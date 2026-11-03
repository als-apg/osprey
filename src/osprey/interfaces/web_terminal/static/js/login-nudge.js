// @ts-check
/* OSPREY Web Terminal — personal-login nudge.
 *
 * A one-time, dismissable card shown when the deployment lets a user run on
 * a personal Claude subscription (`claude_code.allow_personal_login`) and
 * this terminal holds no completed `/login` yet. Both facts come off
 * `GET /api/panels` (`personal_login.allowed`, `personal_login.present`),
 * decided server-side in the terminal's own container; the browser never
 * guesses either. Dismissal is storage-scoped per user, as the tour's is, so
 * users sharing a browser on a multi-user deployment do not dismiss each
 * other's card. A terminal that already holds a login shows nothing.
 *
 * Text is assembled with createElement/textContent — never innerHTML.
 */
import { scopedStorageKey } from '/design-system/js/storage-scope.js';

/** Dismissal flag ("Got it"). Scoped per user. */
const DISMISS_BASE = 'osprey-login-nudge-dismissed-v1';
export const NUDGE_ID = 'login-nudge';
/** Delay between boot and the card, so the shell settles (after the tour's invite). */
export const NUDGE_DELAY_MS = 1200;

const dismissKey = () => scopedStorageKey(DISMISS_BASE);

export const isDismissed = () => {
  try {
    return localStorage.getItem(dismissKey()) === '1';
  } catch {
    return false;
  }
};

/** @param {boolean} on */
export const setDismissed = (on) => {
  try {
    if (on) localStorage.setItem(dismissKey(), '1');
    else localStorage.removeItem(dismissKey());
  } catch {
    /* storage unavailable — the card just shows again next time */
  }
};

/**
 * @param {string} tag @param {string} [cls] @param {string} [text]
 * @returns {HTMLElement}
 */
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

/** Remove the card, if shown. */
export function hideLoginNudge() {
  document.getElementById(NUDGE_ID)?.remove();
}

/**
 * Build and show the card. Idempotent: a card already on the page stays.
 * @returns {HTMLElement}
 */
export function showLoginNudge() {
  const existing = document.getElementById(NUDGE_ID);
  if (existing) return existing;
  const card = el('aside', 'login-nudge');
  card.id = NUDGE_ID;
  card.setAttribute('role', 'status');
  card.setAttribute('aria-labelledby', `${NUDGE_ID}-title`);
  const title = el('h3', 'login-nudge-title', 'Have a personal Claude subscription?');
  title.id = `${NUDGE_ID}-title`;
  const body = el('p', 'login-nudge-body');
  body.append(
    document.createTextNode('Run '),
    el('code', '', '/login'),
    document.createTextNode(
      ' in this terminal and your usage bills to your own account instead of the shared one. The login stays in your own terminal.'
    )
  );
  const actions = el('div', 'login-nudge-actions');
  const gotIt = el('button', 'tour-btn primary', 'Got it');
  gotIt.setAttribute('type', 'button');
  gotIt.onclick = () => {
    setDismissed(true);
    hideLoginNudge();
  };
  const later = el('button', 'tour-btn', 'Not now');
  later.setAttribute('type', 'button');
  later.onclick = () => hideLoginNudge();
  actions.append(gotIt, later);
  card.append(title, body, actions);
  document.body.appendChild(card);
  return card;
}

/**
 * Decide from the `/api/panels` payload whether the card is due.
 * @param {any} panelsPayload
 * @returns {boolean}
 */
export function loginNudgeDue(panelsPayload) {
  const facts = panelsPayload?.personal_login;
  if (!facts || facts.allowed !== true || facts.present !== false) return false;
  if (isDismissed()) return false;
  return !document.body.classList.contains('embedded');
}

/**
 * Show the card when it is due, after the shell has settled.
 * @param {any} panelsPayload
 * @returns {boolean} whether a card was scheduled
 */
export function applyLoginNudge(panelsPayload) {
  if (!loginNudgeDue(panelsPayload)) return false;
  setTimeout(showLoginNudge, NUDGE_DELAY_MS);
  return true;
}
