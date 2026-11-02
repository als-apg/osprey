// @ts-check
/* OSPREY Web Terminal — Sign out, asked first
 *
 * Signing out ends this login's terminal and agent sessions on the server
 * (routes/websocket.py's `logout_terminal` empties both pools), so a stray
 * click must not do it. Every way in — the sidebar's Sign out row, the OSPREY
 * mark when the deployment offers sign-out, and the command palette's
 * "Log out" — goes through {@link confirmSignOut}, which asks with the shared
 * confirm dialog (posture-confirm.js) and only then runs logout.js's flow.
 *
 * Cancel has focus when the dialog opens: the safe answer is the one a stray
 * Enter gives. Escape is the same as Cancel.
 */

import { logout } from './logout.js';
import { dismissConfirm, isConfirmUp, showConfirm } from './posture-confirm.js';

/**
 * Ask whether to sign out; on yes, log out and leave for the landing page.
 *
 * @param {{landingUrl: string, button?: HTMLButtonElement | null}} options
 *   `landingUrl` is where logout.js sends the operator; `button` is the
 *   control that was clicked, if it was a button, so logout.js's in-flight
 *   lock lands on what the operator is looking at.
 * @returns {void}
 */
export function confirmSignOut({ landingUrl, button = null }) {
  /** @param {KeyboardEvent} event */
  const onKey = (event) => {
    if (event.key === 'Escape' && isConfirmUp()) {
      event.preventDefault();
      dismissConfirm();
    }
  };
  showConfirm({
    title: 'Sign out of OSPREY?',
    body: [['Your terminal and agent sessions end. Anything the agent is doing stops.']],
    live: null,
    confirmLabel: 'Sign out',
    focus: 'cancel',
    onConfirm: ({ done }) => {
      done();
      void logout(landingUrl, button);
    },
    onDismiss: () => document.removeEventListener('keydown', onKey, true),
  });
  document.addEventListener('keydown', onKey, true);
}
