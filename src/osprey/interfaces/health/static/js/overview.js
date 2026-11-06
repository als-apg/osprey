// @ts-check
/*
 * System Health dashboard — the overview: ring, headline, scan line and the
 * layer bands with their component tiles.
 */

import { fmtAge, fmtClock, headline, worstSev } from './helpers.js';
import { poll } from './actions.js';
import { h, led, must, now, svg } from './dom.js';
import { openTile, toggleBand, toggleTile } from './drawer.js';
import {
  DEFAULT_INTERVAL_S,
  GAUGE_C,
  GAUGE_R,
  MCP_ID,
  compRows,
  layerRows,
  mcpTileState,
  store,
  tileFact,
  tileSev,
} from './state.js';

export function renderRing() {
  const host = must('ring');
  host.textContent = '';
  const s = svg('svg', { viewBox: '0 0 200 200', class: 'ring-svg', 'aria-hidden': 'true' });
  const defs = svg('defs', {});
  const glow = svg('filter', { id: 'glow', x: '-30%', y: '-30%', width: '160%', height: '160%' });
  glow.appendChild(svg('feGaussianBlur', { stdDeviation: 3, result: 'b' }));
  const merge = svg('feMerge', {});
  merge.appendChild(svg('feMergeNode', { in: 'b' }));
  merge.appendChild(svg('feMergeNode', { in: 'SourceGraphic' }));
  glow.appendChild(merge);
  const bglow = svg('filter', {
    id: 'bglow',
    x: '-200%',
    y: '-200%',
    width: '500%',
    height: '500%',
  });
  bglow.appendChild(svg('feGaussianBlur', { stdDeviation: 2.4 }));
  defs.append(glow, bglow);
  s.appendChild(defs);
  s.appendChild(svg('circle', { class: 'ring-bg', cx: 100, cy: 100, r: GAUGE_R }));

  const counts = store.LAYERS.map((l) =>
    l.components.reduce((a, c) => a + compRows(c).length + (c.id === MCP_ID ? 1 : 0), 0)
  );
  const tot = counts.reduce((a, b) => a + b, 0) || 1;
  let off = 0;
  const gap = 3;
  store.LAYERS.forEach((l, i) => {
    const len = (GAUGE_C * counts[i]) / tot;
    if (!counts[i]) return;
    const sev = worstSev(l.components.map(tileSev));
    const seg = svg('circle', {
      class: 'seg s-' + sev,
      cx: 100,
      cy: 100,
      r: GAUGE_R,
      'stroke-dasharray': Math.max(len - gap, 1) + ' ' + GAUGE_C,
      'stroke-dashoffset': -off,
      transform: 'rotate(-90 100 100)',
      filter: 'url(#glow)',
    });
    const title = svg('title', {});
    title.textContent = l.label + ' · ' + counts[i] + ' checks · ' + sev;
    seg.appendChild(title);
    seg.addEventListener('click', () => toggleBand(l.id));
    s.appendChild(seg);
    off += len;
  });
  s.appendChild(svg('circle', { class: 'ring-orbit', cx: 100, cy: 100, r: 92 }));
  const beam = svg('g', { class: 'beam' });
  beam.appendChild(
    svg('circle', { class: 'beam-halo', cx: 100, cy: 8, r: 6, filter: 'url(#bglow)' })
  );
  beam.appendChild(svg('circle', { class: 'beam-dot', cx: 100, cy: 8, r: 3 }));
  s.appendChild(beam);
  host.appendChild(s);

  const hl = headline(store.LAYERS, layerRows());
  const top = hl.major
    ? [hl.major, 'MAJOR', 'MAJOR', hl.minor + ' MINOR']
    : hl.minor
      ? [hl.minor, 'MINOR', 'MINOR', '0 MAJOR']
      : [0, 'MAJOR · MINOR', 'OK', ''];
  host.appendChild(
    h(
      'div',
      { class: 'ctr' },
      h('div', { class: 'big s-' + top[2], text: top[0] }),
      h('div', { class: 'lab s-' + top[2], text: top[1] }),
      h('div', { class: 'lab2', text: top[3] })
    )
  );
}

export function renderHeadline() {
  const hl = headline(store.LAYERS, layerRows());
  const p = must('headline');
  p.textContent = '';
  p.append(
    h('span', { class: hl.minor ? 'mi' : 'nil', text: hl.minor + ' MINOR' }),
    ' · ',
    h('span', { class: hl.major ? 'ma' : 'nil', text: hl.major + ' MAJOR' })
  );
  // An MCP failure found by introspection is a fault the poll does not carry.
  const faults = hl.faults.slice();
  const ms = mcpTileState();
  if (ms.issue && !faults.some((f) => f.id === MCP_ID))
    faults.unshift({ id: MCP_ID, name: 'MCP servers', issue: ms.issue, sev: 'MAJOR' });
  for (const f of faults.slice(0, 4)) {
    p.append(
      ' — ',
      h('a', {
        href: '#' + encodeURIComponent(f.id),
        class: 's-' + f.sev,
        text: f.name,
        on: {
          click: (/** @type {Event} */ e) => {
            e.preventDefault();
            openTile(f.id);
          },
        },
      }),
      ': ' + f.issue
    );
  }
  if (faults.length > 4) p.append(' — +' + (faults.length - 4) + ' more');
}

export function renderScan() {
  const d = store.env;
  const line = must('scaninfo');
  line.textContent = '';
  if (!d || d.warming) {
    line.append(h('span', { text: 'first scan in progress' }));
    return;
  }
  line.append(
    'scan ',
    h('b', { text: fmtClock(d.scanned_at) }),
    ' · ',
    h('b', { id: 'scanage', text: d.scanned_at ? fmtAge(now() - d.scanned_at) : '—' }),
    ' · ' +
      Math.round(d.elapsed_ms || 0) +
      ' ms · every ' +
      Math.round(d.interval_s || DEFAULT_INTERVAL_S) +
      ' s'
  );
  if (d.stale) line.append(' · ', h('b', { class: 's-MINOR', text: 'stale' }));
  if (d.deadline_hit) line.append(' · ', h('b', { class: 's-MINOR', text: 'deadline hit' }));
}

export function renderBands() {
  const host = must('bands');
  host.textContent = '';
  let first = true;
  for (const l of store.LAYERS) {
    const sev = worstSev(l.components.map(tileSev));
    const tiles = l.components.map((c) => {
      const s = tileSev(c);
      const btn = h(
        'button',
        {
          class: 't s-' + s,
          id: 'tile-' + c.id,
          type: 'button',
          tabindex: first ? 0 : -1,
          'aria-expanded': 'false',
          title: c.name + ' · ' + s,
          data: { tile: c.id },
          on: { click: () => toggleTile(c.id) },
        },
        h('span', { class: 'n' }, led(s), h('span', { class: 'nm', text: c.name })),
        h('span', { class: 'f', text: tileFact(c) })
      );
      first = false;
      return btn;
    });
    host.append(
      h(
        'div',
        {
          class: 'band' + (sev === 'OK' || sev === 'DISABLED' ? ' calm' : ''),
          id: 'band-' + l.id,
          data: { band: l.id },
        },
        h(
          'button',
          {
            class: 'bl s-' + sev,
            type: 'button',
            title: 'all ' + l.label + ' checks',
            data: { bandlabel: l.id },
            on: { click: () => toggleBand(l.id) },
          },
          l.label
        ),
        h('div', { class: 'tiles' }, tiles)
      ),
      h(
        'div',
        { class: 'drawer', id: 'drawer-' + l.id },
        h('div', { class: 'dwrap' }, h('div', { class: 'din' }))
      )
    );
  }
}

export function renderWarming() {
  must('ring').textContent = '';
  const p = must('headline');
  p.textContent = 'first scan in progress';
  renderScan();
  const host = must('bands');
  host.textContent = '';
  host.append(
    h('div', { class: 'ld' }, h('i', { class: 'spin' }), h('span', { text: 'scanning' }))
  );
}

/** @param {string} message */
export function renderError(message) {
  const host = must('bands');
  host.textContent = '';
  host.append(
    h(
      'div',
      { class: 'err-s' },
      h('div', { class: 'em', text: 'checks endpoint unreachable' }),
      h('div', { text: message }),
      h('button', { class: 'btn', type: 'button', text: 'retry', on: { click: poll } })
    )
  );
}
