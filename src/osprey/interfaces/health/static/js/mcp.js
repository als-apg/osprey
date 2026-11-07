// @ts-check
/*
 * System Health dashboard — the MCP servers drawer: introspection on open,
 * the server table, server info and the tool detail (description, arguments,
 * raw schema).
 */

import { rowsBlock } from './rows.js';

import { argTree, firstLine, fmtDur, mdBlocks, tagOf } from './helpers.js';
import { enterClick, h, led, pulse } from './dom.js';
import { drawerHead, openTile, show } from './drawer.js';
import { renderDrill, tagEl } from './drill.js';
import { renderHeadline, renderRing } from './overview.js';
import {
  MCP_ID,
  S,
  ds,
  indexMcp,
  kidsOf,
  mcp,
  mcpTileState,
  store,
  tileFact,
  tileSev,
} from './state.js';
/** @typedef {import('./state.js').Comp} Comp */
/** @typedef {import('./state.js').DrillNode} DrillNode */

/** @param {Comp} c @param {string} sev */
export function mcpCard(c, sev) {
  const st = mcpTileState();
  const head = drawerHead(sev, c.name, st.issue || st.fact, () => probeServer(''), !!st.issue);
  /** @type {(Node | null)[]} */
  const body = [];
  if ((c.rows || []).length) body.push(...rowsBlock(c.rows || []));
  if (mcp.loading && !mcp.data) {
    const n = ((store.env && store.env.mcp && store.env.mcp.configured) || []).length;
    body.push(
      h(
        'div',
        { class: 'meta wait' },
        h('i', { class: 'spin' }),
        ' introspecting ' + n + (n === 1 ? ' server' : ' servers')
      )
    );
  } else if (mcp.error && !mcp.data) {
    body.push(
      h(
        'div',
        { class: 'row s-MAJOR' },
        led('MAJOR'),
        h('span', { class: 'w', text: 'introspection' }),
        h('span', { class: 'v', text: mcp.error }),
        h('span', { class: 'm', text: 'MCP initialize + tools/list' }),
        h(
          'span',
          { class: 'hov' },
          h('button', {
            class: 'rp',
            type: 'button',
            text: 'retry',
            on: { click: () => loadMcp(true) },
          })
        )
      )
    );
  } else if (mcp.data) {
    const st2 = ds(MCP_ID);
    const input = /** @type {HTMLInputElement} */ (
      h('input', {
        type: 'search',
        placeholder: 'filter tools',
        'aria-label': 'filter tools',
        value: st2.q,
      })
    );
    input.addEventListener('input', () => {
      st2.q = input.value.trim();
      if (
        st2.path.some((k) => {
          const nd = store.NODES[k];
          const p = nd && store.NODES[nd.parent];
          return !nd || !p || !kidsOf(p, st2.q).includes(nd);
        })
      ) {
        st2.path = [];
        st2.col = 0;
      }
      renderDrill(MCP_ID);
    });
    body.push(h('div', { class: 'mtool' }, input));
    body.push(h('div', { class: 'drill', id: 'drill-' + MCP_ID }));
  }
  return [h('div', { class: 'caret' }), h('div', { class: 'dcard' }, head, ...body)];
}

/** The server table: tools · startup · list. @param {boolean} go Rows open the tile at that server. */
export function serverTable(go) {
  const servers = mcp.data ? mcp.data.servers || [] : [];
  const tab = h(
    'div',
    { class: 'stab' },
    h(
      'div',
      { class: 'sh' },
      h('span'),
      h('span'),
      h('span'),
      h('span', { text: 'tools', title: 'tools returned by tools/list' }),
      h('span', { text: 'startup', title: 'process spawn + MCP initialize (cold start)' }),
      h('span', { text: 'list', title: 'tools/list round trip' })
    )
  );
  for (const s of servers) {
    const key = MCP_ID + '/' + s.name;
    const n = store.NODES[key];
    const sev = n ? n.sev : 'DISABLED';
    const http = s.transport === 'http';
    const open = () =>
      go
        ? openTile(MCP_ID, [key])
        : ((ds(MCP_ID).path = [key]), (ds(MCP_ID).col = 0), renderDrill(MCP_ID, true));
    const cells = http
      ? [
          h('span', { class: 'num', text: '—' }),
          h('span', { class: 'num', text: '—' }),
          h('span', { class: 'num', text: '—' }),
        ]
      : [
          h('span', { class: 'num', text: Array.isArray(s.tools) ? String(s.tools.length) : '—' }),
          h('span', { class: 'num', text: fmtDur(s.init_ms) }),
          h('span', { class: 'num', text: fmtDur(s.list_ms) }),
        ];
    tab.append(
      h(
        'div',
        {
          class: 'srow s-' + sev,
          role: 'button',
          tabindex: 0,
          title: http
            ? 'transport http; the stdio probe lists no tools'
            : [s.command, ...(s.args || [])].join(' '),
          on: { click: open, keydown: enterClick },
        },
        h('span', { class: 'cv', text: http ? '' : '›' }),
        led(sev),
        h(
          'span',
          { class: 'nm' },
          h('span', { text: s.name }),
          http
            ? h('span', { class: 'note', text: 'http · not probed' })
            : !Array.isArray(s.tools)
              ? h('span', { class: 'note s-MAJOR', text: s.error || 'no answer' })
              : null
        ),
        ...cells
      )
    );
  }
  return tab;
}

/** @param {DrillNode} n */
export function serverInfo(n) {
  const s = n.srv;
  if (s.transport === 'http')
    return h('div', {
      class: 'meta',
      text: 'http server' + (s.url ? ' · ' + s.url : '') + ' · tools not listed by the stdio probe',
    });
  const busy = mcp.busy.has(s.name);
  /** @type {(Node | null)[]} */
  const out = [];
  if (!Array.isArray(s.tools)) {
    out.push(
      h(
        'div',
        { class: 'row s-MAJOR' },
        led('MAJOR'),
        h('span', { class: 'w', text: 'initialize' }),
        h('span', { class: 'v', text: s.error || 'no answer' }),
        h('span', { class: 'm', text: 'spawn + MCP initialize' })
      )
    );
  } else {
    out.push(
      h(
        'div',
        { class: 'line' + (busy ? ' busy' : '') },
        s.tools.length + ' tools · startup ' + fmtDur(s.init_ms) + ' · list ' + fmtDur(s.list_ms),
        h(
          'span',
          { class: 'hov' },
          busy
            ? h('i', { class: 'spin' })
            : h('button', {
                class: 'rp',
                type: 'button',
                text: 're-probe',
                on: { click: () => probeServer(s.name) },
              })
        )
      )
    );
  }
  out.push(
    h('div', {
      class: 'meta',
      title: [s.command, ...(s.args || [])].join(' '),
      text: [s.server_version, s.protocol ? 'protocol ' + s.protocol : '', s.transport || 'stdio']
        .filter(Boolean)
        .join(' · '),
    })
  );
  if (s.instructions)
    out.push(h('div', { class: 'instr', title: s.instructions, text: firstLine(s.instructions) }));
  return h('div', null, ...out);
}

/** @param {DrillNode} n */
export function toolDetail(n) {
  const t = n.tool;
  const schema = t.inputSchema || {};
  const tag = tagOf(t);
  const desc = h('div', { class: 'desc' });
  for (const b of mdBlocks(t.description)) {
    if (b.kind === 'p') desc.append(h('p', null, spansEl(b.spans)));
    else {
      if (b.lead.length) desc.append(h('p', null, spansEl(b.lead)));
      desc.append(
        h(
          'ul',
          null,
          b.items.map((it) => h('li', null, spansEl(it)))
        )
      );
    }
  }
  const raw = h('pre', { class: 'raw', hidden: true, text: JSON.stringify(schema, null, 2) });
  const rawBtn = h('button', { class: 'rawt', type: 'button', text: 'raw schema' });
  rawBtn.addEventListener('click', () => {
    raw.hidden = !raw.hidden;
    rawBtn.textContent = raw.hidden ? 'raw schema' : 'hide raw schema';
  });
  const tree = argTree(schema);
  const args = tree.length
    ? h(
        'div',
        { class: 'args' },
        h(
          'div',
          { class: 'ar h' },
          h('span', { text: 'name', title: '* = required' }),
          h('span', { text: 'type' }),
          h('span', { text: 'default' })
        ),
        ...argRowsEl(tree, 0)
      )
    : h('div', { class: 'meta0', text: 'none' });
  return h(
    'div',
    { class: 'tooldet' },
    h('h3', null, h('span', { text: t.name }), tag.tag ? tagEl(tag) : null),
    desc,
    h('div', { class: 'sech' }, h('span', { text: 'arguments' }), rawBtn),
    args,
    raw
  );
}

/** @param {{code: boolean, text: string}[]} spans */
export function spansEl(spans) {
  return spans.map((s) => (s.code ? h('code', { text: s.text }) : document.createTextNode(s.text)));
}

/**
 * @param {import('./helpers.js').ArgNode[]} nodes
 * @param {number} depth
 * @returns {HTMLElement[]}
 */
export function argRowsEl(nodes, depth) {
  /** @type {HTMLElement[]} */
  const out = [];
  for (const a of nodes) {
    const sub = a.children.length
      ? h('div', { class: 'argsub', hidden: true }, ...argRowsEl(a.children, depth + 1))
      : null;
    const toggle = sub
      ? h('button', { class: 'argt', type: 'button', 'aria-expanded': 'false', text: '▸' })
      : null;
    if (toggle && sub) {
      toggle.addEventListener('click', () => {
        sub.hidden = !sub.hidden;
        toggle.textContent = sub.hidden ? '▸' : '▾';
        toggle.setAttribute('aria-expanded', String(!sub.hidden));
      });
    }
    const an = h(
      'span',
      { class: 'an d' + Math.min(depth, 4) },
      h(
        'span',
        { class: 'anm' },
        toggle,
        a.name,
        a.required ? h('span', { class: 'req', title: 'required', text: '*' }) : null
      ),
      a.desc ? h('span', { class: 'ax', title: a.desc, text: a.desc }) : null
    );
    out.push(
      h(
        'div',
        { class: 'ar' },
        an,
        h('span', { class: 'at', text: a.type }),
        h('span', { class: 'ad', text: a.def })
      )
    );
    if (sub) out.push(sub);
  }
  return out;
}

/** @param {boolean} [force] */
export async function loadMcp(force) {
  if (mcp.loading || (mcp.data && !force)) return;
  mcp.loading = true;
  mcp.error = '';
  refreshMcpViews();
  try {
    mcp.data = force ? await store.api.mcpProbe() : await store.api.mcp();
  } catch (e) {
    mcp.error = e instanceof Error ? e.message : String(e);
  } finally {
    mcp.loading = false;
  }
  indexMcp();
  refreshMcpViews();
}

/** @param {string} server Empty = every server. */
export async function probeServer(server) {
  if (!mcp.data) return loadMcp(true);
  const names = server
    ? [server]
    : (mcp.data.servers || [])
        .filter((/** @type {any} */ s) => s.transport !== 'http')
        .map((/** @type {any} */ s) => s.name);
  for (const n of names) mcp.busy.add(n);
  const btn = /** @type {HTMLButtonElement | null} */ (
    document.querySelector('.din [data-probeall]')
  );
  if (btn && !server) btn.disabled = true;
  refreshMcpViews();
  try {
    mcp.data = await store.api.mcpProbe(server || undefined);
  } catch (e) {
    mcp.error = e instanceof Error ? e.message : String(e);
  } finally {
    for (const n of names) mcp.busy.delete(n);
  }
  indexMcp();
  refreshMcpViews(names);
}

/** Re-render whatever shows MCP data: the tile, the headline, an open MCP or band drawer. @param {string[]} [flash] */
export function refreshMcpViews(flash) {
  if (!store.env) return;
  renderRing();
  renderHeadline();
  const t = store.TILES[MCP_ID];
  if (t) {
    const btn = document.getElementById('tile-' + MCP_ID);
    if (btn) {
      const s = tileSev(t.comp);
      btn.className = 't s-' + s + (S.key === MCP_ID ? ' sel' : '');
      const f = btn.querySelector('.f');
      if (f) f.textContent = tileFact(t.comp);
      const l = btn.querySelector('.led');
      if (l) l.className = 'led s-' + s;
    }
  }
  if (S.key === MCP_ID || (t && S.key === 'band:' + t.band)) {
    show(/** @type {string} */ (S.band), /** @type {string} */ (S.key), undefined, true);
    if (flash)
      for (const n of flash)
        document.querySelectorAll('.din .line, .din .srow').forEach((e) => {
          if (e.textContent && e.textContent.includes(n)) pulse(/** @type {HTMLElement} */ (e));
        });
  }
}
