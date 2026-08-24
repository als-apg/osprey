/* OSPREY Web Terminal — Agent Settings post-save notice */

import { mountOverlay, fadeOutOverlay } from './modal-overlay.js';

/**
 * Show the post-save notice: the settings warning dialog's own modal surface,
 * with one acknowledging button and no decision to make. It reuses the
 * `.settings-warning-*` classes drawer.css already styles, so this panel has
 * one modal look rather than two.
 *
 * Shown when a successful config write comes back carrying a `detail`. Today
 * exactly one thing puts it there: a read-only render zone, where the write to
 * `config.yml` landed but the `.claude/` artifacts derived from it are
 * root-owned and only re-render when the container restarts. Without this the
 * operator sees an ordinary "saved" and reasonably concludes a render-shaping
 * edit is already in effect.
 *
 * A modal rather than the panel's own status strip -- which is the idiom the
 * scaffold gallery uses for the same class of message -- because a successful
 * save *closes the drawer* on its way to restarting the terminal. A notice
 * written into the strip would be carried off screen by the close, unread.
 *
 * Deliberately not a gate: it acknowledges something that already happened, so
 * it blocks nothing, resolves nothing, and never writes the per-session
 * acknowledgment the drawer's warning gate keys on. Saving is not
 * acknowledging, and the next drawer open must still warn.
 *
 * Its own module rather than a few more lines in settings.js: that file sits at
 * the 450-line cap eslint puts on production interface JS, which is exactly the
 * signal to split rather than to raise the cap.
 *
 * @param {string} message  the server's own detail, rendered verbatim
 */
export function showSettingsNotice(message) {
  const overlay = document.createElement('div');
  overlay.className = 'settings-warning-overlay';
  overlay.innerHTML = `
    <div class="settings-warning-dialog">
      <div class="settings-warning-icon">ⓘ</div>
      <div class="settings-warning-title">Saved — restart required</div>
      <div class="settings-warning-body"><p></p></div>
      <div class="settings-warning-actions">
        <button class="settings-warning-proceed">OK</button>
      </div>
    </div>
  `;
  // Assigned as text, never interpolated into the markup above: this string
  // arrives over the wire. It is the server's own copy today, and a save notice
  // is still not the place to hand a response body to an HTML parser.
  /** @type {HTMLElement} */ (overlay.querySelector('.settings-warning-body p')).textContent = message;

  mountOverlay(overlay);

  // The same teardown the warning dialog uses: fade out, remove when the
  // transition ends, and a timer in case the transition never fires.
  const dismiss = () => {
    document.removeEventListener('keydown', onKey);
    fadeOutOverlay(overlay);
  };
  /** @type {HTMLElement} */ (overlay.querySelector('.settings-warning-proceed')).addEventListener('click', dismiss);
  /** @param {KeyboardEvent} e */
  const onKey = (e) => { if (e.key === 'Escape') dismiss(); };
  document.addEventListener('keydown', onKey);
}
