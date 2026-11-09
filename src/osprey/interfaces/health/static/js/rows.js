// @ts-check
/*
 * System Health dashboard — check rows: LED · label · value · method, problem
 * rows first, the fix line after them, hover actions and busy/flash state.
 */

import { copyText } from '/design-system/js/clipboard.js';
import { commonMethod } from './helpers.js';
import { ping, reprobe } from './actions.js';
import { h, led, pulse } from './dom.js';
import { rerenderDrawer } from './drawer.js';
import { PING_COMPONENTS, busyRows, rowView, store } from './state.js';

/** @param {number} i @param {boolean} nom Suppress the method (the group row shows it). */
export function rowEl(i, nom) {
  const r = store.R[i];
  const v = rowView(i);
  const busy = busyRows.has(r.name);
  const canPing = r.category === 'providers' && PING_COMPONENTS.has(r.component || '');
  const btn = busy
    ? h('i', { class: 'spin', 'aria-label': 'probing' })
    : h('button', {
        class: 'rp',
        type: 'button',
        text: canPing ? 'ping' : 're-probe',
        on: {
          click: (/** @type {Event} */ e) => {
            e.stopPropagation();
            canPing ? ping(r.name) : reprobe({ checks: [r.name] }, [r.name]);
          },
        },
      });
  const tip = [v.full, r.details, v.method, r.name].filter(Boolean).join('\n');
  return h(
    'div',
    { class: 'row s-' + v.sev + (busy ? ' busy' : ''), data: { name: r.name }, title: tip },
    led(v.sev),
    h('span', { class: 'w', text: v.label }),
    h('span', { class: 'v', text: v.value }),
    h('span', { class: 'm', text: nom ? '' : v.method }),
    h('span', { class: 'hov' }, btn)
  );
}

/** @param {string} fix */
export function fixEl(fix) {
  const b = h('button', { class: 'rp', type: 'button', text: 'copy' });
  b.addEventListener('click', async (e) => {
    e.stopPropagation();
    const ok = await copyText(fix);
    b.textContent = ok ? 'copied' : 'copy failed';
    setTimeout(() => {
      b.textContent = 'copy';
    }, 1200);
  });
  return h(
    'div',
    { class: 'fix' },
    h('code', { text: '$ ' + fix }),
    h('span', { class: 'hov' }, b)
  );
}

/** @param {number[]} idx */
export const methodsOf = (idx) => idx.map((i) => rowView(i).method);

/**
 * Rows grouped problem-first, the fix line after the problems.
 *
 * @param {number[]} idx
 * @param {string | null} [fix]
 */
export function rowsBlock(idx, fix) {
  const nom = !!commonMethod(methodsOf(idx));
  const by = (/** @type {string[]} */ sevs) => idx.filter((i) => sevs.includes(rowView(i).sev));
  const prob = by(['MAJOR', 'MINOR']);
  const fixes = new Set(prob.map((i) => store.R[i].fix).filter(Boolean));
  if (prob.length && fix) fixes.add(fix);
  return [
    ...prob.map((i) => rowEl(i, nom)),
    ...[...fixes].map((f) => fixEl(/** @type {string} */ (f))),
    ...by(['OK']).map((i) => rowEl(i, nom)),
    ...by(['DISABLED']).map((i) => rowEl(i, nom)),
  ];
}

/** @param {number[]} idx @param {string} [cls] */
export function methodTag(idx, cls) {
  const m = commonMethod(methodsOf(idx));
  if (!m) return null;
  return h('span', { class: cls || 'gm', text: m, title: [...new Set(methodsOf(idx))].join('\n') });
}

/** Swap busy markers on rows in place, without a re-render. @param {string[]} names */
export function paintBusy(names) {
  for (const n of names) {
    document.querySelectorAll('.row[data-name="' + CSS.escape(n) + '"]').forEach((row) => {
      const busy = busyRows.has(n);
      row.classList.toggle('busy', busy);
      const hov = row.querySelector('.hov');
      if (busy && hov) {
        hov.textContent = '';
        hov.append(h('i', { class: 'spin', 'aria-label': 'probing' }));
      }
    });
  }
  if (!names.some((n) => busyRows.has(n))) rerenderDrawer(true);
}

/** @param {string[]} names */
export function flashRows(names) {
  for (const n of names)
    document
      .querySelectorAll('.row[data-name="' + CSS.escape(n) + '"]')
      .forEach((e) => pulse(/** @type {HTMLElement} */ (e)));
}
