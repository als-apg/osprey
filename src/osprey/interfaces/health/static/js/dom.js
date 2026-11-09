// @ts-check
/*
 * System Health dashboard — DOM builders and small UI effects.
 *
 * Server strings reach the DOM only through textContent / attribute setters
 * here; nothing in the panel assigns innerHTML.
 */

import { SVG_NS } from './state.js';

/**
 * Build an element. Props: class, text, title, data (object), on (object of
 * listeners); any other key is an attribute (null/false skipped).
 *
 * @param {string} tag
 * @param {Record<string, any> | null} [props]
 * @param {...(Node | string | null | undefined | false | (Node | string | null | undefined | false)[])} kids
 * @returns {HTMLElement}
 */
export function h(tag, props, ...kids) {
  const n = document.createElement(tag);
  if (props) {
    for (const k in props) {
      const v = props[k];
      if (v == null || v === false) continue;
      if (k === 'class') n.className = v;
      else if (k === 'text') n.textContent = String(v);
      else if (k === 'data') for (const d in v) n.dataset[d] = String(v[d]);
      else if (k === 'on') for (const ev in v) n.addEventListener(ev, v[ev]);
      else n.setAttribute(k, v === true ? '' : String(v));
    }
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    n.appendChild(typeof kid === 'string' ? document.createTextNode(kid) : kid);
  }
  return n;
}

/**
 * @param {string} tag
 * @param {Record<string, string | number>} attrs
 */
export function svg(tag, attrs) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) n.setAttribute(k, String(attrs[k]));
  return n;
}

/** @param {string} id */
export function must(id) {
  const node = document.getElementById(id);
  if (!node) throw new Error('missing element: ' + id);
  return node;
}

/** @param {string} sev */
export const led = (sev) => h('span', { class: 'led s-' + sev });

export const now = () => Date.now() / 1000;

/** @param {KeyboardEvent} e */
export function enterClick(e) {
  if ((e.key === 'Enter' || e.key === ' ') && e.target === e.currentTarget) {
    e.preventDefault();
    /** @type {HTMLElement} */ (e.currentTarget).click();
  }
}

/** @param {HTMLElement} e */
export function pulse(e) {
  e.classList.remove('flash');
  void e.offsetWidth;
  e.classList.add('flash');
}

/** A transient line in the scan row (refusals, failures). @param {string} text */
export function note(text) {
  const n = must('note');
  n.textContent = text;
  n.hidden = false;
  setTimeout(() => {
    n.hidden = true;
  }, 6000);
}
