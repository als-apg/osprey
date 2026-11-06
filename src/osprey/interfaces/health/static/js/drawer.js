// @ts-check
/*
 * System Health dashboard — the in-place drawer under a band: tile and band
 * cards, open/move/close, caret, scroll-into-view, probe age and deep links.
 */

import { commonMethod, fmtAge, formatHash, mcpFact, parseHash, worstSev } from './helpers.js';
import { reprobe } from './actions.js';
import { enterClick, h, led, now } from './dom.js';
import { renderDrill } from './drill.js';
import { loadMcp, mcpCard, serverTable } from './mcp.js';
import { methodTag, methodsOf, rowEl, rowsBlock } from './rows.js';
import {
  DS,
  MCP_ID,
  RANK,
  S,
  UNUSED_ID,
  UNUSED_NAME,
  chainOf,
  compRows,
  ds,
  mcp,
  rowView,
  store,
  tileSev,
} from './state.js';
/** @typedef {import('./state.js').Comp} Comp */
/** @typedef {import('./state.js').DrillNode} DrillNode */

/**
 * @param {string} sev
 * @param {string} title
 * @param {string} sub
 * @param {() => void} onProbeAll
 * @param {boolean} [issue] The sub line is the fault text (colored by severity).
 */
export function drawerHead(sev, title, sub, onProbeAll, issue) {
  return h(
    'div',
    { class: 'dh' },
    led(sev),
    h('h2', { text: title }),
    sub
      ? h('span', {
          class: 'sub' + (issue && (sev === 'MINOR' || sev === 'MAJOR') ? ' s-' + sev : ''),
          text: sub,
          title: sub,
        })
      : null,
    h('span', { class: 'sp' }),
    h('span', { class: 'pa' }),
    h('button', {
      class: 'rp',
      type: 'button',
      text: 're-probe all',
      data: { probeall: '1' },
      on: { click: onProbeAll },
    }),
    h('button', {
      class: 'x',
      type: 'button',
      'aria-label': 'close',
      text: '×',
      on: { click: closeDrawer },
    })
  );
}

/** @param {string[] | undefined} used */
export const usedEl = (used) =>
  used && used.length ? h('div', { class: 'usedby', text: 'used by ' + used.join(', ') }) : null;

/** @param {Comp} c */
export function tileCard(c) {
  const sev = tileSev(c);
  if (c.id === MCP_ID) return mcpCard(c, sev);
  const groups = c.groups || [];
  const all = compRows(c);
  const isIssue = !c.header && (sev === 'MINOR' || sev === 'MAJOR') && !!c.issue;
  const sub = c.header || (isIssue ? c.issue || '' : c.fact || '');
  const head = drawerHead(
    sev,
    c.name,
    sub,
    () =>
      reprobe(
        { component: c.id },
        all.map((i) => store.R[i].name)
      ),
    isIssue
  );
  /** @type {(Node | null)[]} */
  const body = [];
  if (groups.length) {
    if ((c.rows || []).length) body.push(...rowsBlock(c.rows || []));
    body.push(h('div', { class: 'drill', id: 'drill-' + c.id }));
  } else {
    // Rows sharing one probe method drop it; it shows once above them.
    const shared = methodTag(c.rows || [], 'gmi');
    if (shared) body.push(h('div', { class: 'gml' }, shared));
    body.push(...rowsBlock(c.rows || []));
    if ((c.unused || []).length) {
      body.push(h('div', { class: 'grph' }, UNUSED_NAME, methodTag(c.unused || [], 'gf')));
      const nom = !!commonMethod(methodsOf(c.unused || []));
      body.push(...(c.unused || []).map((i) => rowEl(i, nom)));
    }
  }
  return [
    h('div', { class: 'caret' }),
    h('div', { class: 'dcard' }, head, usedEl(c.used_by), ...body),
  ];
}

/** @param {string} bandId */
export function bandCard(bandId) {
  const l = store.LAYERS.find((x) => x.id === bandId);
  if (!l) return [];
  const sev = worstSev(l.components.map(tileSev));
  const all = l.components.flatMap(compRows);
  const order = [...l.components].sort(
    (a, b) => RANK[/** @type {'OK'} */ (tileSev(b))] - RANK[/** @type {'OK'} */ (tileSev(a))]
  );
  /** @type {Node[]} */
  const body = [];
  for (const c of order) {
    const s = tileSev(c);
    const groups = c.groups || [];
    body.push(
      h(
        'div',
        { class: 'grph' },
        led(s),
        h('span', { text: c.name }),
        c.header
          ? h('span', { class: 'gf', text: c.header })
          : groups.length
            ? null
            : methodTag(c.rows || [], 'gf')
      )
    );
    if (c.id === MCP_ID) {
      body.push(
        mcp.data
          ? serverTable(true)
          : h('div', {
              class: 'meta',
              text: mcpFact(store.env && store.env.mcp) + ' · tools listed when the tile opens',
            })
      );
    } else if (groups.length) {
      body.push(...rowsBlock(c.rows || []));
      for (const g of groups) body.push(goItem(c.id + '/' + g.id));
      if ((c.unused || []).length) body.push(goItem(c.id + '/' + UNUSED_ID));
    } else {
      body.push(...rowsBlock(c.rows || []));
      if ((c.unused || []).length) {
        body.push(h('div', { class: 'grph sub' }, UNUSED_NAME, methodTag(c.unused || [], 'gf')));
        body.push(...(c.unused || []).map((i) => rowEl(i, true)));
      }
    }
  }
  return [
    h('div', { class: 'caret' }),
    h(
      'div',
      { class: 'dcard' },
      drawerHead(sev, l.label, '', () =>
        reprobe(
          { checks: all.map((i) => store.R[i].name) },
          all.map((i) => store.R[i].name)
        )
      ),
      ...body
    ),
  ];
}

/** A group listed in a band drawer: clicking opens its tile at that group. @param {string} key */
export function goItem(key) {
  const n = store.NODES[key];
  if (!n) return h('span');
  const tileId = key.split('/')[0];
  return h(
    'div',
    {
      class: 'it s-' + n.sev,
      role: 'button',
      tabindex: 0,
      title: n.name,
      on: { click: () => openTile(tileId, chainOf(key)), keydown: enterClick },
    },
    h('span', { class: 'cv', text: '›' }),
    led(n.sev),
    h('span', { class: 'nm', text: n.name }),
    h('span', { class: 'sm', text: n.summary || '' })
  );
}

/** @param {string} key */
export function drawerSignature(key) {
  if (key.startsWith('band:')) {
    const l = store.LAYERS.find((x) => 'band:' + x.id === key);
    return l ? JSON.stringify([l, l.components.flatMap(compRows).map((i) => store.R[i])]) : '';
  }
  const t = store.TILES[key];
  return t ? JSON.stringify([t.comp, compRows(t.comp).map((i) => store.R[i])]) : '';
}

/**
 * Open (or re-render) a drawer.
 *
 * @param {string} bandId
 * @param {string} key Tile id or 'band:<id>'.
 * @param {string[]} [path] Drill path to select.
 * @param {boolean} [inPlace] Re-render without animation or scrolling.
 */
export function show(bandId, key, path, inPlace) {
  const same = S.band === bandId;
  if (S.band && !same) {
    const old = document.getElementById('drawer-' + S.band);
    if (old) {
      old.classList.remove('open');
      const oi = old.querySelector('.din');
      setTimeout(() => {
        if (oi && !old.classList.contains('open')) oi.textContent = '';
      }, 220);
    }
  }
  S.band = bandId;
  S.key = key;
  const tile = store.TILES[key];
  if (tile && path && path.length) {
    const st = ds(key);
    st.path = path;
    st.col = Math.max(0, path.length - 1);
  }
  const dr = document.getElementById('drawer-' + bandId);
  if (!dr) return;
  const din = /** @type {HTMLElement} */ (dr.querySelector('.din'));
  din.textContent = '';
  din.append(...(key.startsWith('band:') ? bandCard(bandId) : tile ? tileCard(tile.comp) : []));
  store.drawerSig = drawerSignature(key);
  markSel();
  if (!same) requestAnimationFrame(() => dr.classList.add('open'));
  else dr.classList.add('open');
  if (tile && store.NODES[key]) renderDrill(key);
  updAge();
  placeCaret();
  setHash();
  if (!inPlace) setTimeout(keepInView, same ? 0 : 230);
  if (key === MCP_ID && !mcp.data && !mcp.loading && !mcp.error) loadMcp(false);
}

/** Re-render the open drawer in place when its data changed (or when forced). @param {boolean} [force] */
export function rerenderDrawer(force) {
  if (!S.key || !S.band) return;
  if (!S.key.startsWith('band:') && !store.TILES[S.key]) {
    closeDrawer();
    return;
  }
  const sig = drawerSignature(S.key);
  if (!force && sig === store.drawerSig) {
    updAge();
    return;
  }
  const ae = document.activeElement;
  const focusSearch = ae instanceof HTMLInputElement && ae.type === 'search';
  show(S.band, S.key, undefined, true);
  if (focusSearch) {
    const inp = /** @type {HTMLInputElement | null} */ (
      document.querySelector('.din input[type=search]')
    );
    if (inp) {
      inp.focus();
      inp.setSelectionRange(inp.value.length, inp.value.length);
    }
  }
}

export function closeDrawer() {
  if (!S.band) return;
  const dr = document.getElementById('drawer-' + S.band);
  const k = S.key;
  if (dr) {
    dr.classList.remove('open');
    const din = dr.querySelector('.din');
    setTimeout(() => {
      if (din && !dr.classList.contains('open')) din.textContent = '';
    }, 220);
  }
  S.band = null;
  S.key = null;
  placed = { band: null, key: null, left: 0 };
  store.drawerSig = '';
  markSel();
  setHash();
  const el = k && !k.startsWith('band:') ? document.getElementById('tile-' + k) : null;
  if (el) el.focus({ preventScroll: true });
}

/** @param {string} id @param {string[]} [path] */
export function openTile(id, path) {
  const t = store.TILES[id];
  if (t) show(t.band, id, path);
}

/** @param {string} id */
export function toggleTile(id) {
  if (S.key === id) closeDrawer();
  else openTile(id);
}

/** @param {string} id */
export function toggleBand(id) {
  if (S.key === 'band:' + id) closeDrawer();
  else show(id, 'band:' + id);
}

export function markSel() {
  document.querySelectorAll('.t.sel, .bl.sel').forEach((e) => {
    e.classList.remove('sel');
    e.setAttribute('aria-expanded', 'false');
  });
  document.querySelectorAll('.band.open').forEach((e) => e.classList.remove('open'));
  if (!S.key || !S.band) return;
  const band = document.getElementById('band-' + S.band);
  if (band) band.classList.add('open');
  const el = selTarget();
  if (el) {
    el.classList.add('sel');
    el.setAttribute('aria-expanded', 'true');
  }
}

export function selTarget() {
  if (!S.key || !S.band) return null;
  return S.key.startsWith('band:')
    ? document.querySelector('[data-bandlabel="' + CSS.escape(S.band) + '"]')
    : document.getElementById('tile-' + S.key);
}

/** Where the caret last stood, so the next tile in the same band can slide from it. */
let placed = {
  band: /** @type {string | null} */ (null),
  key: /** @type {string | null} */ (null),
  left: 0,
};

export function placeCaret() {
  if (!S.band) return;
  const dr = document.getElementById('drawer-' + S.band);
  const car = /** @type {HTMLElement | null} */ (dr && dr.querySelector('.caret'));
  const target = selTarget();
  if (!dr || !car || !target) return;
  const x = target.getBoundingClientRect();
  const d = dr.getBoundingClientRect();
  const left = x.left + x.width / 2 - d.left;
  // Slide only from the previous tile of this same open drawer; every other
  // placement (fresh open, other band, poll re-render, resize) just lands.
  const from = placed.band === S.band && placed.key !== S.key ? placed.left : left;
  car.style.transition = 'none';
  car.style.left = from + 'px';
  void car.offsetWidth;
  car.style.transition = '';
  car.style.left = left + 'px';
  placed = { band: S.band, key: S.key, left };
}

export function keepInView() {
  if (!S.band) return;
  const band = document.getElementById('band-' + S.band);
  const dr = document.getElementById('drawer-' + S.band);
  if (!band || !dr) return;
  const top = band.getBoundingClientRect().top;
  const bot = dr.getBoundingClientRect().bottom;
  const vh = window.innerHeight;
  if (top < 0) {
    band.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    return;
  }
  if (bot > vh) {
    const dy = Math.min(bot - vh + 8, top - 8);
    if (dy > 0) document.body.scrollBy({ top: dy, behavior: 'smooth' });
  }
}

/** Probe age once per drawer: the oldest probe among its rows. */
export function updAge() {
  const scan = document.getElementById('scanage');
  if (scan && store.env && store.env.scanned_at)
    scan.textContent = fmtAge(now() - store.env.scanned_at);
  const pa = document.querySelector('.din .pa');
  if (!pa || !S.key) return;
  /** @type {number[]} */
  let ts = [];
  if (S.key === MCP_ID) {
    if (mcp.data && mcp.data.probed_at) ts = [mcp.data.probed_at];
  } else {
    const idx = S.key.startsWith('band:')
      ? (
          store.LAYERS.find((x) => 'band:' + x.id === S.key) || { components: [] }
        ).components.flatMap(compRows)
      : store.TILES[S.key]
        ? compRows(store.TILES[S.key].comp)
        : [];
    ts = idx.map((i) => rowView(i).t).filter(Boolean);
  }
  pa.textContent = ts.length
    ? 'probed ' + fmtAge(now() - Math.min(...ts))
    : mcp.loading && S.key === MCP_ID
      ? 'probing'
      : '';
}

export function setHash() {
  /** @type {string[]} */
  let parts = [];
  if (S.key && S.band) {
    if (S.key.startsWith('band:')) parts = ['band-' + S.band];
    else {
      const st = DS[S.key];
      parts = [S.key, ...(st ? st.path.map((k) => k.split('/').pop() || '') : [])];
    }
  }
  const h2 = formatHash(parts);
  if (h2 !== location.hash)
    history.replaceState(null, '', location.pathname + location.search + h2);
}

export function fromHash() {
  const parts = parseHash(location.hash);
  if (!parts.length) return;
  if (parts[0].startsWith('band-') && store.LAYERS.some((l) => l.id === parts[0].slice(5))) {
    show(parts[0].slice(5), 'band:' + parts[0].slice(5));
    return;
  }
  const t = store.TILES[parts[0]];
  if (!t) return;
  /** @type {string[]} */
  const path = [];
  let key = parts[0];
  for (const p of parts.slice(1)) {
    key += '/' + p;
    path.push(key);
  }
  if (parts[0] === MCP_ID && path.length) {
    // The servers are introspected on open; keep the path until they arrive.
    store.pendingMcpPath = path;
  }
  openTile(
    parts[0],
    path.filter((k) => store.NODES[k])
  );
}
