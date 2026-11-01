// @ts-check
/* OSPREY Design System — Pop-out client
 *
 * The popup side of the web terminal's pop-out protocol (the hub side is the
 * web terminal's popout.js). A panel page the hub opened in its own window
 * reports to the hub so the hub can retire the panel's tile while the window
 * is open, keep the window's theme and session in step, and put
 * the tile back when the window goes away.
 *
 * The page knows it is a pop-out from its address (`?popout=true`,
 * frame-params.js), and which panel it is from its window name: the hub opens
 * it as `osprey-popout-<panelId>`, and a window keeps its name across every
 * navigation inside it. The hub is `window.opener`.
 *
 *   on load      `osprey-popout-hello {panelId}` to the opener
 *   on pagehide  `osprey-popout-bye {panelId}` to the opener
 *   roll-call    the hub posts `osprey-popout-roll-call` on the BroadcastChannel
 *                after a reload (its own state is gone, the opener link is
 *                not); the answer is another hello
 *
 * Every message goes to the opener only, addressed to this page's own origin,
 * and only while that opener is alive. A page whose opener is gone — closed,
 * or the address pasted into a fresh tab — is a plain standalone page and
 * sends nothing. The hub's state messages (`osprey-theme-change`,
 * `osprey-session-change`) arrive by the same channel every embedded frame uses
 * and are applied by the receivers those pages already run (theme-manager.js,
 * the page's own session receiver), each of which takes a message from this
 * origin only.
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';

/** The BroadcastChannel the hub rolls call on, scoped like a storage key. */
const CHANNEL = 'osprey-popout';

/** The window name the hub opens a pop-out under, followed by its panel id. */
const WINDOW_NAME_PREFIX = 'osprey-popout-';

const HELLO = 'osprey-popout-hello';
const BYE = 'osprey-popout-bye';
const ROLL_CALL = 'osprey-popout-roll-call';

/**
 * The panel id this window was popped out for, read off the window name, or
 * null when the name is not the protocol's.
 * @returns {string | null}
 */
export function popoutPanelId() {
  const name = window.name;
  if (typeof name !== 'string' || !name.startsWith(WINDOW_NAME_PREFIX)) return null;
  const id = name.slice(WINDOW_NAME_PREFIX.length);
  return id === '' ? null : id;
}

/**
 * The hub this window answers to: the opener, while it is still open.
 * @returns {Window | null}
 */
function hub() {
  try {
    const opener = window.opener;
    return opener && !opener.closed ? opener : null;
  } catch {
    return null;
  }
}

/**
 * Report to the hub: hello now, bye on `pagehide`, hello again on every
 * roll-call. Called by frame-params.js's applyEmbedded() on a pop-out page.
 * @returns {boolean} whether this page reports to a hub
 */
export function initPopoutClient() {
  const panelId = popoutPanelId();
  if (panelId === null || hub() === null) return false;

  /** @param {string} type */
  const post = (type) => {
    const target = hub();
    if (!target) return;
    try {
      target.postMessage({ type, panelId }, window.location.origin);
    } catch {
      /* the opener navigated out of reach — there is no hub to report to */
    }
  };

  post(HELLO);
  window.addEventListener('pagehide', () => post(BYE));
  const channel = new BroadcastChannel(scopedStorageKey(CHANNEL));
  channel.addEventListener('message', (event) => {
    if (event.data && event.data.type === ROLL_CALL) post(HELLO);
  });
  return true;
}
