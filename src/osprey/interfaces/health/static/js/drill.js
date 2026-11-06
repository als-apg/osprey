// @ts-check
/*
 * System Health dashboard — Miller-column drill-down inside a drawer: groups
 * (or MCP servers › tools) in columns, the selected node's detail on the right.
 */

import { tagOf } from './helpers.js';
import { reprobe } from './actions.js';
import { h, led } from './dom.js';
import { setHash } from './drawer.js';
import { probeServer, serverInfo, serverTable, toolDetail } from './mcp.js';
import { methodTag, rowsBlock } from './rows.js';
import { MCP_ID, NARROW_MILL_PX, ds, kidsOf, mcp, store } from './state.js';
/** @typedef {import('./state.js').DrillNode} DrillNode */

/** @param {string} tileId @param {boolean} [focus] */
export function renderDrill(tileId, focus) {
  const host = document.getElementById('drill-' + tileId);
  const root = store.NODES[tileId];
  if (!host || !root) return;
  const st = ds(tileId);
  const q = st.q;
  /** @type {{c: number, k: DrillNode[]}[]} */
  const cols = [];
  /** @type {DrillNode} */
  let n = root;
  for (let c = 0; ; c++) {
    const k = kidsOf(n, q);
    if (!k.length) break;
    cols.push({ c, k });
    const sel = st.path[c];
    if (!sel || !store.NODES[sel]) break;
    n = store.NODES[sel];
  }
  const last = st.path.length ? store.NODES[st.path[st.path.length - 1]] : null;
  const narrow = host.clientWidth > 0 && host.clientWidth < NARROW_MILL_PX;
  const railN = narrow && cols.length >= 2 ? cols.length - 1 : 0;
  const tpl =
    cols.map((_, i) => (i < railN ? '30px' : 'fit-content(250px)')).join(' ') + ' minmax(0,1fr)';
  const mill = h('div', { class: 'mill', tabindex: 0 });
  mill.style.gridTemplateColumns = tpl;
  cols.forEach((col, i) => {
    if (i < railN) {
      const s = store.NODES[st.path[i]];
      mill.append(
        h(
          'div',
          {
            class: 'mcol rail',
            title: s ? s.name : '',
            on: {
              click: () => {
                st.path = st.path.slice(0, i);
                st.col = i;
                renderDrill(tileId, true);
              },
            },
          },
          h('span', { text: s ? s.name : '' })
        )
      );
      return;
    }
    mill.append(
      h(
        'div',
        { class: 'mcol' + (st.col === i ? ' act' : '') },
        col.k.map((k) => colItem(tileId, k, i, st.path[i] === k.key))
      )
    );
  });
  mill.append(
    h(
      'div',
      { class: 'mcol detail' },
      last ? detailOf(last) : tileId === MCP_ID ? serverTable(false) : null
    )
  );
  mill.addEventListener('keydown', (e) => millKey(e, tileId));
  host.textContent = '';
  host.append(mill);
  if (focus) mill.focus({ preventScroll: true });
  const sel = /** @type {HTMLElement | null} */ (mill.querySelector('.mcol.act .it.sel'));
  if (sel && sel.parentElement) {
    const c = sel.parentElement;
    if (
      sel.offsetTop < c.scrollTop ||
      sel.offsetTop + sel.offsetHeight > c.scrollTop + c.clientHeight
    ) {
      c.scrollTop = sel.offsetTop - c.clientHeight / 2;
    }
  }
  setHash();
}

/**
 * @param {string} tileId
 * @param {DrillNode} k
 * @param {number} col
 * @param {boolean} sel
 */
export function colItem(tileId, k, col, sel) {
  const tag = k.kind === 'tool' ? tagOf(k.tool) : null;
  const probe =
    k.kind === 'group' && (k.rows || []).length
      ? h(
          'span',
          { class: 'hov' },
          h('button', {
            class: 'rp',
            type: 'button',
            text: 're-probe',
            on: {
              click: (/** @type {Event} */ e) => {
                e.stopPropagation();
                const names = (k.rows || []).map((i) => store.R[i].name);
                reprobe({ checks: names }, names);
              },
            },
          })
        )
      : k.kind === 'server' && k.srv.transport !== 'http'
        ? h(
            'span',
            { class: 'hov' },
            mcp.busy.has(k.name)
              ? h('i', { class: 'spin' })
              : h('button', {
                  class: 'rp',
                  type: 'button',
                  text: 're-probe',
                  on: {
                    click: (/** @type {Event} */ e) => {
                      e.stopPropagation();
                      probeServer(k.name);
                    },
                  },
                })
          )
        : null;
  return h(
    'div',
    {
      class: 'it ' + k.kind + ' s-' + k.sev + (sel ? ' sel' : ''),
      role: 'button',
      tabindex: -1,
      title: k.name,
      data: { key: k.key },
      on: {
        click: () => {
          const st = ds(tileId);
          st.path = st.path.slice(0, col).concat(k.key);
          st.col = col;
          renderDrill(tileId, true);
        },
      },
    },
    h('span', { class: 'cv', text: k.children.length ? '›' : '' }),
    k.kind === 'tool' ? h('span') : led(k.sev),
    h('span', { class: 'nm' }, h('span', { text: k.name }), tag && tag.tag ? tagEl(tag) : null),
    probe
  );
}

/** @type {Record<string, string>} */
const TAG_SOURCE = {
  hooks: "from the render's hook_config.json",
  annotations: 'from the tool annotations; hook_config.json unreadable',
  name: 'from the tool name; no hook_config.json and no annotations',
};

/** @param {{tag: string, derived: boolean, from?: string}} tag */
export function tagEl(tag) {
  return h('span', {
    class: 'tag',
    text: tag.tag,
    title: TAG_SOURCE[tag.from || (tag.derived ? 'name' : 'annotations')],
  });
}

/** @param {DrillNode} n */
export function detailOf(n) {
  if (n.kind === 'tool') return toolDetail(n);
  if (n.kind === 'server') return serverInfo(n);
  const rows = n.rows || [];
  return h(
    'div',
    null,
    h(
      'div',
      { class: 'dhd' },
      h('span', { class: 'dn', text: n.name }),
      h('span', { class: 'ds s-' + n.sev, text: n.summary || '' }),
      methodTag(rows, 'gmi')
    ),
    ...rowsBlock(rows, n.fix)
  );
}

/**
 * @param {KeyboardEvent} e
 * @param {string} tileId
 */
export function millKey(e, tileId) {
  const st = ds(tileId);
  const q = st.q;
  const parentOf = (/** @type {number} */ c) =>
    c === 0 ? store.NODES[tileId] : store.NODES[st.path[c - 1]];
  const parent = parentOf(st.col);
  const list = parent ? kidsOf(parent, q) : [];
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    e.stopPropagation();
    if (!list.length) return;
    let i = list.findIndex((k) => k.key === st.path[st.col]);
    i = i < 0 ? 0 : Math.max(0, Math.min(list.length - 1, i + (e.key === 'ArrowDown' ? 1 : -1)));
    st.path = st.path.slice(0, st.col).concat(list[i].key);
    renderDrill(tileId, true);
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    e.stopPropagation();
    const cur = st.path[st.col] ? store.NODES[st.path[st.col]] : null;
    if (!cur) {
      if (list.length) {
        st.path = st.path.slice(0, st.col).concat(list[0].key);
        renderDrill(tileId, true);
      }
      return;
    }
    const k = kidsOf(cur, q);
    if (k.length) {
      st.col++;
      st.path = st.path.slice(0, st.col).concat(k[0].key);
      renderDrill(tileId, true);
    }
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault();
    e.stopPropagation();
    if (st.col > 0) {
      st.path = st.path.slice(0, st.col);
      st.col--;
      renderDrill(tileId, true);
    }
  }
}
