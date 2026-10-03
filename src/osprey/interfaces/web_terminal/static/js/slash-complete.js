// @ts-check
/* OSPREY Web Terminal — slash-command suggestions for the Simple view.
 *
 * Attaches a suggestion list to the console's message box. The contract:
 *
 * - The list is active exactly when the whole message is a single token that
 *   begins with `/`. The text after the slash is the query; rows are the
 *   commands that fuzzy-match it, best first, ties by name. An empty query
 *   lists every command by name. No rows means the list is closed.
 * - The first row is armed after every render. ArrowDown and ArrowUp move the
 *   armed row, wrapping at either end.
 * - Tab, or Enter without Shift, accepts the armed row; a click accepts the
 *   clicked row. Accepting only fills the box with `/<name> ` and never sends:
 *   the text the operator sends is always the text in the box.
 * - Escape closes the list until the text changes.
 * - A key the open list does not take, any key while it is closed, and any key
 *   during an IME composition is left to the caller untouched.
 * - Every transition from closed to active reloads the commands; the previous
 *   list is shown while the reload is in flight, and a failed reload keeps it.
 */

import { el } from '/design-system/js/dom.js';
import { fuzzyMatch } from '/design-system/js/fuzzy.js';

/** A message the list is active for: one token beginning with `/`. */
const SLASH_TOKEN = /^\/\S*$/;

let nextListboxId = 0;

/**
 * Attach slash-command suggestions to *textarea*.
 * @param {HTMLTextAreaElement} textarea
 * @param {{ load: () => Promise<import('./chat-client.js').SlashCommand[]>, mount: HTMLElement }} opts
 * @returns {{ handleKeydown: (e: KeyboardEvent) => boolean, close: () => void, isOpen: () => boolean }}
 */
export function attachSlashComplete(textarea, { load, mount }) {
  const listbox = el('div', 'op-slash-popup');
  listbox.id = `op-slash-listbox-${nextListboxId++}`;
  listbox.setAttribute('role', 'listbox');
  listbox.hidden = true;
  mount.append(listbox);

  textarea.setAttribute('role', 'combobox');
  textarea.setAttribute('aria-autocomplete', 'list');
  textarea.setAttribute('aria-controls', listbox.id);
  textarea.setAttribute('aria-expanded', 'false');

  /** The last list loaded, or null before any load has succeeded. */
  let commands = /** @type {import('./chat-client.js').SlashCommand[] | null} */ (null);

  /** The commands currently shown, in row order. */
  let shown = /** @type {import('./chat-client.js').SlashCommand[]} */ ([]);

  /** Index of the armed row in `shown`. */
  let armed = 0;

  /** Whether the message is a slash token the list has not been closed on. */
  let active = false;

  /** The value Escape closed the list on; an input with it stays closed. */
  let dismissedAt = /** @type {string | null} */ (null);

  const isOpen = () => !listbox.hidden;

  /** Close the list; the next active input reloads the commands. */
  function close() {
    active = false;
    hide();
  }

  function hide() {
    listbox.hidden = true;
    listbox.replaceChildren();
    shown = [];
    textarea.setAttribute('aria-expanded', 'false');
    textarea.removeAttribute('aria-activedescendant');
  }

  /** @param {string} query */
  function rank(query) {
    const all = commands ?? [];
    if (!query) return [...all].sort((a, b) => a.name.localeCompare(b.name));
    return all
      .map((command) => ({ command, match: fuzzyMatch(query, command.name) }))
      .filter((entry) => entry.match !== null)
      .sort(
        (a, b) =>
          /** @type {{ score: number }} */ (b.match).score -
            /** @type {{ score: number }} */ (a.match).score ||
          a.command.name.localeCompare(b.command.name)
      )
      .map((entry) => entry.command);
  }

  /** @param {number} index */
  function arm(index) {
    const rows = listbox.children;
    for (let i = 0; i < rows.length; i++) {
      const on = i === index;
      rows[i].setAttribute('aria-selected', on ? 'true' : 'false');
      rows[i].classList.toggle('armed', on);
    }
    armed = index;
    textarea.setAttribute('aria-activedescendant', `${listbox.id}-${index}`);
  }

  function render() {
    shown = rank(textarea.value.slice(1));
    if (shown.length === 0) {
      hide();
      return;
    }
    const rows = shown.map((command, i) => {
      const row = el('div', 'op-slash-row');
      row.id = `${listbox.id}-${i}`;
      row.setAttribute('role', 'option');
      const name = el('span', 'op-slash-name');
      name.textContent = `/${command.name}`;
      row.append(name);
      if (command.argument_hint) {
        const hint = el('span', 'op-slash-hint');
        hint.textContent = command.argument_hint;
        row.append(hint);
      }
      const desc = el('span', 'op-slash-desc');
      desc.textContent = command.description;
      row.append(desc);
      // Keep focus in the message box: the click accepts, the press must not blur.
      row.addEventListener('mousedown', (e) => e.preventDefault());
      row.addEventListener('click', () => accept(i));
      return row;
    });
    listbox.replaceChildren(...rows);
    listbox.hidden = false;
    textarea.setAttribute('aria-expanded', 'true');
    arm(0);
  }

  function reload() {
    load().then(
      (list) => {
        commands = list;
        if (active) render();
      },
      () => {}
    );
  }

  function onInput() {
    const value = textarea.value;
    if (dismissedAt !== null && value !== dismissedAt) dismissedAt = null;
    const now = SLASH_TOKEN.test(value) && dismissedAt === null;
    const fresh = now && !active;
    active = now;
    if (!active) {
      hide();
      return;
    }
    if (fresh) reload();
    render();
  }

  /** @param {number} index */
  function accept(index) {
    const command = shown[index];
    if (!command) return;
    const value = `/${command.name} `;
    textarea.value = value;
    textarea.setSelectionRange(value.length, value.length);
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    textarea.focus();
  }

  /** @param {KeyboardEvent} e */
  function handleKeydown(e) {
    if (!isOpen() || e.isComposing) return false;
    const count = shown.length;
    switch (e.key) {
      case 'ArrowDown':
        arm((armed + 1) % count);
        break;
      case 'ArrowUp':
        arm((armed - 1 + count) % count);
        break;
      case 'Tab':
        accept(armed);
        break;
      case 'Enter':
        if (e.shiftKey) return false;
        accept(armed);
        break;
      case 'Escape':
        dismissedAt = textarea.value;
        close();
        break;
      default:
        return false;
    }
    e.preventDefault();
    return true;
  }

  textarea.addEventListener('input', onInput);
  textarea.addEventListener('blur', close);

  return { handleKeydown, close, isOpen };
}
