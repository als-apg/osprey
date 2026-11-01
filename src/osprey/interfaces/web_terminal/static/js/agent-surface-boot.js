// @ts-check
/* OSPREY Web Terminal — Agent surface boot
 *
 * Stamps html[data-agent-surface="terminal"] before first paint when this tab
 * opted into the terminal (agent-surface.js owns the state), so a reload never
 * flashes the chat first. A page with data-terminal="off" stays in chat.
 *
 * Deliberately NOT an ES module: it must run synchronously in <head>, so it
 * imports nothing and mirrors storage-scope.js's key rule inline.
 */
(function () {
  'use strict';

  try {
    const root = document.documentElement;
    if (root.getAttribute('data-terminal') === 'off') return;
    const scope = root.getAttribute('data-osprey-storage-scope');
    const key = 'osprey-agent-surface' + (scope ? '--' + scope : '');
    if (window.sessionStorage.getItem(key) === 'terminal') {
      root.setAttribute('data-agent-surface', 'terminal');
    }
  } catch {
    /* storage blocked: the tile opens on the chat */
  }
})();
