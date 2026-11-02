// @ts-check
/* OSPREY Web Terminal — which surface the Agent tile shows
 *
 * The Agent tile shows the operator chat. A tab may opt into the terminal at
 * run time, from the last row of the tile's ⋯ menu (sessions.js); the choice is
 * `html[data-agent-surface="terminal"]`, and an absent attribute means chat.
 *
 * The choice lives in sessionStorage, so a reload keeps it and a new tab starts
 * in chat. agent-surface-boot.js stamps it before first paint and mirrors the
 * key below; CSS gates the two containers off the attribute (terminal.css,
 * operator.css), so a change tears down no DOM.
 *
 * This module holds the state and says when it changes. What a change means for
 * the live agent is app.js's follow-up: one agent runs a session at a time, so
 * the session is handed to the surface now showing.
 *
 * @module agent-surface
 */

import { terminalOffered } from '/design-system/js/frame-params.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';

const ATTRIBUTE = 'data-agent-surface';
const STORAGE_KEY = 'osprey-agent-surface';
const TERMINAL = 'terminal';

/** @type {Set<() => void>} */
const listeners = new Set();

/**
 * Whether the Agent tile is showing the terminal. A page that offers no
 * terminal (`web.terminal: false`) never does.
 * @returns {boolean}
 */
export function isTerminalSurface() {
  return document.documentElement.getAttribute(ATTRIBUTE) === TERMINAL && terminalOffered();
}

/**
 * Call `fn` after every change of surface. The attribute is already current.
 * @param {() => void} fn
 */
export function onSurfaceChange(fn) {
  listeners.add(fn);
}

/** Show the terminal on the Agent tile. Does nothing where it is not offered. */
export function showTerminalSurface() {
  if (!terminalOffered() || isTerminalSurface()) return;
  document.documentElement.setAttribute(ATTRIBUTE, TERMINAL);
  store(TERMINAL);
  for (const fn of listeners) fn();
}

/** Show the operator chat on the Agent tile. */
export function showChatSurface() {
  if (!isTerminalSurface()) return;
  document.documentElement.removeAttribute(ATTRIBUTE);
  store(null);
  for (const fn of listeners) fn();
}

/** @param {string | null} value */
function store(value) {
  try {
    const key = scopedStorageKey(STORAGE_KEY);
    if (value) window.sessionStorage.setItem(key, value);
    else window.sessionStorage.removeItem(key);
  } catch {
    /* storage blocked: the surface still applies until this page goes away */
  }
}
