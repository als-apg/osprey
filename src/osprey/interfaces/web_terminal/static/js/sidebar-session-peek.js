// @ts-check
/* OSPREY Web Terminal — the session tree's peek
 *
 * Hovering a folded session row for a moment shows that session's artifacts
 * in a popover beside the sidebar (or beside the collapsed sidebar's flyout),
 * so they can be seen, and opened, without switching to the session. The
 * peek is the design system's popover skin (`.osprey-popover`), placed here
 * rather than through popover.js: that module moves focus into what it opens
 * and closes every other popover, which a hover must not do.
 *
 *   - OPEN. 300 ms after the pointer comes to rest on a folded session row
 *     (`aria-expanded="false"`), once that session's listing has loaded and
 *     holds at least one artifact. An expanded session lists its artifacts
 *     in the tree already, so it gets no peek.
 *   - STAY. Moving the pointer from the row into the peek keeps it open: the
 *     close waits 150 ms, and entering the peek cancels it.
 *   - CLOSE. On leaving both the row and the peek, on Escape, on a press
 *     outside the peek, on a scroll outside it, and when the row unfolds or
 *     leaves the tree.
 *   - CLICK. An artifact row opens that artifact (`onPick`, the tree's own
 *     artifact click); "N more" switches to the session (`onMore`).
 *
 * The rows are the tree's own (sidebar-artifact-rows.js), from the session's
 * listing (sidebar-artifact-listing.js), newest first and at most
 * {@link PEEK_ROWS}. A session title names the peek in its
 * `aria-controls` (sidebar-artifact-rows.js), so the tooltip layer stays
 * quiet while the peek shows in its place.
 *
 * @module sidebar-session-peek
 */

import * as rows from './sidebar-artifact-rows.js';

/** @typedef {import('./sidebar-artifact-rows.js').TreeEntry} TreeEntry */

/** The peek's element id, named by every folded session title's `aria-controls`. */
export const PEEK_ID = rows.SESSION_PEEK_ID;

/** Rest on a folded session this long before its peek opens. */
export const PEEK_OPEN_MS = 300;

/** Leave the row and the peek for this long before the peek closes. */
export const PEEK_CLOSE_MS = 150;

/** The most artifact rows a peek lists. */
export const PEEK_ROWS = 8;

/** Space between the sidebar's edge and the peek, and the window margin. */
const GAP = 8;

const TREE_SELECTOR = '[data-session-tree]';
const SESSION_SELECTOR = '.osprey-tree-session[data-session-id]';

/**
 * @typedef {object} PeekListing
 * @property {boolean} loaded
 * @property {TreeEntry[]} entries  newest first
 */

/**
 * @typedef {object} PeekDeps
 * @property {(treeId: string) => PeekListing | null} listing  the session's
 *   listing, its first page asked for; null while artifacts are not served
 * @property {(treeId: string) => number | null} total  the session's artifact count
 * @property {() => string | null} shownId  the artifact the viewer shows
 * @property {(id: string) => void} onPick  open an artifact
 * @property {(treeId: string) => void} onMore  switch to the session
 */

/**
 * The rows a peek lists: what the tree lists in Activity mode (a run once, by
 * its head; never a run's parts).
 * @param {TreeEntry[]} entries
 * @returns {TreeEntry[]}
 */
export function peekEntries(entries) {
  return entries.filter(
    (e) => e.form && !(e.exec_id && e.run_part === 'output' && e.form !== 'runs')
  );
}

/**
 * Make the peek. It does nothing until {@link watch} wires the document.
 * @param {PeekDeps} deps
 */
export function createSessionPeek(deps) {
  /** @type {HTMLElement | null} */
  let el = null;
  /** The session the peek is for (set while it waits for its listing too). */
  /** @type {string | null} */
  let openId = null;
  /** The tree the session's row is in. @type {Element | null} */
  let host = null;
  let opening = 0;
  let closing = 0;

  function element() {
    if (el?.isConnected) return el;
    el = document.createElement('div');
    el.id = PEEK_ID;
    el.className = 'osprey-popover sidebar-session-peek';
    el.setAttribute('role', 'dialog');
    el.hidden = true;
    el.addEventListener('click', onPeekClick);
    document.body.append(el);
    return el;
  }

  /** @returns {HTMLElement | null} the open session's row in its tree */
  function anchor() {
    if (!openId || !host?.isConnected) return null;
    for (const row of host.querySelectorAll(SESSION_SELECTOR)) {
      if (row instanceof HTMLElement && row.dataset.sessionId === openId) return row;
    }
    return null;
  }

  /**
   * Start showing a session's peek (it shows once its listing has rows).
   * @param {string} treeId @param {Element} tree
   */
  function open(treeId, tree) {
    clearTimeout(closing);
    openId = treeId;
    host = tree;
    refresh();
  }

  /** Close the peek now, and drop a peek waiting to open. */
  function close() {
    clearTimeout(opening);
    dismiss();
  }

  /**
   * Close the open peek. A peek waiting to open stays scheduled: the pointer
   * that moved from one folded session onto the next has asked for it.
   */
  function dismiss() {
    clearTimeout(closing);
    openId = null;
    host = null;
    if (el && !el.hidden) {
      const hadFocus = el.contains(document.activeElement);
      el.hidden = true;
      el.replaceChildren();
      if (hadFocus) /** @type {HTMLElement | null} */ (document.activeElement)?.blur?.();
    }
  }

  function closeLater() {
    clearTimeout(closing);
    closing = window.setTimeout(dismiss, PEEK_CLOSE_MS);
  }

  /**
   * Redraw the open peek from the session's listing; close it when its row
   * unfolded or left the tree. The tree calls this after every render.
   */
  function refresh() {
    if (!openId) return;
    const row = anchor();
    if (!row || row.getAttribute('aria-expanded') === 'true') return close();
    const listing = deps.listing(openId);
    if (!listing) return close();
    const peek = element();
    const entries = peekEntries(listing.entries);
    if (!listing.loaded || !entries.length) {
      peek.hidden = true;
      return;
    }
    const name = row.querySelector('.osprey-tree-title')?.textContent ?? '';
    const head = document.createElement('div');
    head.className = 'osprey-popover-title';
    head.textContent = name;
    const list = document.createElement('div');
    list.className = 'sidebar-session-peek-list';
    list.setAttribute('role', 'listbox');
    list.setAttribute('aria-label', `Artifacts of ${name}`);
    const shownId = deps.shownId();
    const shown = entries.slice(0, PEEK_ROWS);
    for (const entry of shown) {
      list.append(
        rows.artifactRow(entry, {
          shownId,
          depth: 0,
          iconName: entry.form === 'runs' ? rows.RUN_ICON : undefined,
        })
      );
    }
    /** @type {HTMLElement[]} */
    const kids = [head, list];
    const rest = Math.max(deps.total(openId) ?? 0, entries.length) - shown.length;
    if (rest > 0) {
      const more = document.createElement('button');
      more.type = 'button';
      more.className = 'osprey-tree-more sidebar-session-peek-more';
      more.dataset.peekMore = '';
      more.textContent = `${rest} more`;
      kids.push(more);
    }
    peek.setAttribute('aria-label', `Artifacts of ${name}`);
    peek.replaceChildren(...kids);
    peek.hidden = false;
    place(peek, row);
  }

  /** @param {HTMLElement} peek @param {HTMLElement} row */
  function place(peek, row) {
    const rect = row.getBoundingClientRect();
    const edge =
      row.closest('#sidebar, .osprey-popover')?.getBoundingClientRect().right ?? rect.right;
    peek.style.left = `${Math.round(edge + GAP)}px`;
    const top = Math.min(rect.top, window.innerHeight - peek.offsetHeight - GAP);
    peek.style.top = `${Math.round(Math.max(GAP, top))}px`;
  }

  /** @param {MouseEvent} event */
  function onPeekClick(event) {
    const target = event.target instanceof Element ? event.target : null;
    if (!target || !openId) return;
    const treeId = openId;
    if (target.closest('[data-peek-more]')) {
      close();
      deps.onMore(treeId);
      return;
    }
    const row = target.closest('.osprey-list-row[data-id]');
    if (!(row instanceof HTMLElement) || !row.dataset.id) return;
    close();
    deps.onPick(row.dataset.id);
  }

  /** @param {PointerEvent} event */
  function onOver(event) {
    const target = event.target instanceof Element ? event.target : null;
    if (!target) return;
    if (el?.contains(target)) {
      clearTimeout(closing);
      return;
    }
    clearTimeout(opening);
    const row = target.closest(`${TREE_SELECTOR} ${SESSION_SELECTOR}`);
    const tree = row?.closest(TREE_SELECTOR);
    if (row instanceof HTMLElement && tree && row.getAttribute('aria-expanded') === 'false') {
      const treeId = row.dataset.sessionId ?? '';
      if (treeId === openId) {
        clearTimeout(closing);
        return;
      }
      if (openId) closeLater();
      opening = window.setTimeout(() => {
        if (row.isConnected) open(treeId, tree);
      }, PEEK_OPEN_MS);
      return;
    }
    if (openId) closeLater();
  }

  /** @param {KeyboardEvent} event */
  function onKey(event) {
    if (event.key === 'Escape' && openId) close();
  }

  /** @param {Event} event */
  function onOutside(event) {
    if (!openId) return;
    const target = event.target instanceof Node ? event.target : null;
    if (target && el?.contains(target)) return;
    close();
  }

  /**
   * Wire the document until `signal` aborts.
   * @param {AbortSignal} signal
   */
  function watch(signal) {
    document.addEventListener('pointerover', onOver, { signal });
    document.documentElement.addEventListener('pointerleave', closeLater, { signal });
    document.addEventListener('keydown', onKey, { signal });
    document.addEventListener('pointerdown', onOutside, { capture: true, signal });
    document.addEventListener('scroll', onOutside, { capture: true, passive: true, signal });
    window.addEventListener('blur', close, { signal });
    signal.addEventListener('abort', () => {
      close();
      el?.remove();
      el = null;
    });
  }

  return {
    watch,
    refresh,
    close,
    /** @returns {string | null} the session the peek is open for */
    openFor: () => (el && !el.hidden ? openId : null),
  };
}
