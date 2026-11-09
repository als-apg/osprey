// @ts-check
/* OSPREY Web Terminal — Context menu of the terminal body
 *
 * One place, registered with the shell's place dispatcher (place-menu.js): the
 * xterm body (#terminal-container and everything xterm draws inside it).
 *
 *   Copy · Paste · Select all · Clear terminal · Switch to chat · Restart terminal
 *
 * Copy acts on xterm's OWN selection, the one a Shift-drag (Option-drag on
 * macOS) makes. A plain drag is the agent's selection, which the agent copies
 * itself (OSC 52), so with nothing selected in xterm the row is disabled and
 * says so. Paste goes through `term.paste`, xterm's bracketed paste, so the
 * agent receives it as one paste and not as typed keys.
 *
 * Copy and Paste reach the clipboard through the window the event happened in
 * (`ownerDocument.defaultView.navigator`), so they work while the Agent pane is
 * popped out into its own window.
 *
 * The keyboard route: xterm keeps a single custom key handler, so this module
 * does not attach one. It returns `handleKey`, which terminal.js composes into
 * the handler it already attaches (beside Ctrl+Shift+C). A menu key there opens
 * the menu and is marked `preventDefault`, so the dispatcher's own keydown route
 * does not open it a second time.
 *
 * No import-time side effects: terminal.js installs it from initTerminal.
 */

import { registerPlace } from './place-menu.js';
import { hintFor } from './key-bindings.js';
import { showUndoNotice } from './undo-notice.js';
import { isContextMenuKey, openContextMenuAt } from '/design-system/js/context-menu.js';

/** @typedef {import('/design-system/js/context-menu.js').MenuItem} MenuItem */

/** The menu's accessible name. */
const ARIA_LABEL = 'Terminal';

/** Why Copy is disabled with nothing selected in xterm. */
export const COPY_DISABLED_TIP =
  "Nothing is selected in the terminal. A plain drag is the agent's selection and is copied " +
  'automatically; Shift-drag (Option-drag on macOS) selects in the terminal.';

/** Why Paste is disabled where the page may not read the clipboard. */
export const PASTE_DISABLED_TIP =
  'Paste needs clipboard access, which the browser grants only to a secure (https or localhost) page.';

/** The notice shown when the browser refuses a paste at click time. */
export const PASTE_REFUSED_NOTICE = 'The browser refused to read the clipboard.';

/** The notice shown when the browser refuses a copy at click time. */
export const COPY_REFUSED_NOTICE = 'The browser refused to copy the terminal selection.';

/**
 * @typedef {object} TerminalMenuOptions
 * @property {HTMLElement} container - the element xterm is opened in
 * @property {any} term - the xterm Terminal
 * @property {() => unknown} onRestart - restarts the terminal session
 * @property {() => void} onSwitchToChat - shows the chat surface
 */

/**
 * The window an event happened in, or the container's window.
 * @param {Event | null | undefined} event
 * @param {HTMLElement} container
 * @returns {any}
 */
function viewOf(event, container) {
  const target = /** @type {any} */ (event?.target);
  const doc = target?.ownerDocument ?? container.ownerDocument;
  return doc?.defaultView ?? null;
}

/**
 * The clipboard of `view`, or null.
 * @param {any} view
 */
function clipboardOf(view) {
  return view?.navigator?.clipboard ?? null;
}

/**
 * Whether `view` may read the clipboard: a secure context with the async API.
 * @param {any} view
 * @returns {boolean}
 */
function canPaste(view) {
  return (
    !!view && view.isSecureContext !== false && typeof clipboardOf(view)?.readText === 'function'
  );
}

/**
 * Register the terminal body's place.
 * @param {TerminalMenuOptions} options
 * @returns {{ handleKey: (event: KeyboardEvent) => boolean, uninstall: () => void }}
 *   `handleKey` answers true when it took the event (a menu key), which the
 *   composed xterm handler turns into `false` so xterm does not encode it.
 */
export function installTerminalMenu({ container, term, onRestart, onSwitchToChat }) {
  /**
   * @param {Event | null | undefined} event
   * @returns {MenuItem[]}
   */
  function build(event) {
    const view = viewOf(event, container);
    const doc = view?.document ?? container.ownerDocument;
    const hasSelection = !!term?.hasSelection?.();
    const pasteOk = canPaste(view);
    return /** @type {MenuItem[]} */ ([
      {
        label: 'Copy',
        hint: hintFor('terminal-copy'),
        disabled: !hasSelection,
        ...(hasSelection ? {} : { tip: COPY_DISABLED_TIP }),
        run: () => {
          const text = term.getSelection();
          Promise.resolve()
            .then(() => clipboardOf(view).writeText(text))
            .catch(() => showUndoNotice({ message: COPY_REFUSED_NOTICE, doc }));
        },
      },
      {
        label: 'Paste',
        disabled: !pasteOk,
        ...(pasteOk ? {} : { tip: PASTE_DISABLED_TIP }),
        run: () => {
          Promise.resolve()
            .then(() => clipboardOf(view).readText())
            .then((text) => {
              if (text) term.paste(text);
              term.focus?.();
            })
            .catch(() => showUndoNotice({ message: PASTE_REFUSED_NOTICE, doc }));
        },
      },
      { label: 'Select all', run: () => term.selectAll() },
      { label: 'Clear terminal', run: () => term.clear() },
      { divider: true },
      { label: 'Switch to chat', run: () => onSwitchToChat() },
      { divider: true },
      { label: 'Restart terminal', danger: true, run: () => void onRestart() },
    ]);
  }

  const unregister = registerPlace({
    match: (el) => el === container,
    ariaLabel: ARIA_LABEL,
    build: (_el, event) => build(event),
  });

  /**
   * The menu-key branch of xterm's custom key handler.
   * @param {KeyboardEvent} event
   * @returns {boolean}
   */
  function handleKey(event) {
    if (!isContextMenuKey(event)) return false;
    if (event.type === 'keydown') {
      event.preventDefault();
      openContextMenuAt(container, { ariaLabel: ARIA_LABEL, items: build(event) });
    }
    return true;
  }

  return { handleKey, uninstall: unregister };
}
