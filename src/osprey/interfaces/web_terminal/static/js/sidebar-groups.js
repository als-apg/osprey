// @ts-check
/* OSPREY Web Terminal — the sidebar's groups
 *
 * The sidebar sorts its panel rows into named groups. Two are built in,
 * `workspace` ("Workspace") and `tools` ("Tools"); an operator can rename
 * them, add groups of their own, move any group up or down the sidebar, and
 * delete the groups they added. Every row except the Agent row can move to
 * another group. The Agent row is always the first row of the first group,
 * because the Agent pane is the anchor of the workspace.
 *
 * MODEL. The groups are the operator's own arrangement, so they live in this
 * browser, per user, and never on the server. One JSON value is stored under
 * `scopedStorageKey('osprey-sidebar-groups')`:
 *
 *   { version: 1,
 *     groups: [{ id, label }, …],         in sidebar order; an id is
 *                                         `workspace`, `tools` or `g-<8 hex>`
 *     placement: { <panel id>: <group id> } }
 *
 * A panel with no placement sits where the sidebar has always put it:
 * Artifacts and Events in Workspace, every other panel in Tools. Within a
 * group the rows keep the server's rail order (panel-rail.js), so a group
 * filters the rail and never reorders it. A stored value without this shape
 * reads as the default, and the next change overwrites it. The value is read
 * again whenever it changes, so a change made in another tab of the same user
 * is the base of the next change made here. When storage refuses a write, the
 * change lives in this page's memory for as long as the page does.
 *
 * Every change is stored first and then announced to the `onGroupsChange`
 * listeners; panel-rail.js listens and moves the rows into their groups.
 *
 * HEADINGS. Each group's heading (`.osprey-section-label`) holds its name
 * (`span.sidebar-group-name`, also the group's accessible name) and a ⋯
 * button (`button.osprey-mini[data-group-menu]`, shown on hover and on focus)
 * that opens the group's menu: Rename, New group below, Move up, Move down,
 * and Delete group for a group the operator added. The menu is one
 * body-level `.osprey-popover[role=menu]` (`#sidebar-group-menu`), placed
 * beside the sidebar and dismissed by the design system's popover.js. Rename
 * swaps the name for a text field in place: Enter, or leaving the field,
 * keeps the new name; Escape keeps the old one, and so does an empty name.
 */

import { icon } from '/design-system/js/icons.js';
import { closePopover, openPopover } from '/design-system/js/popover.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { TERMINAL_RAIL_ID } from './panel-catalog.js';

/**
 * @typedef {object} SidebarGroup
 * @property {string} id - `workspace`, `tools`, or `g-` and eight hex digits
 * @property {string} label - what the heading shows: 1 to 40 characters, trimmed
 */

/**
 * @typedef {object} GroupsModel
 * @property {1} version
 * @property {SidebarGroup[]} groups - in sidebar order
 * @property {Record<string, string>} placement - panel id → group id
 */

/** The storage key, scoped per user through storage-scope.js. */
export const GROUPS_STORAGE_KEY = 'osprey-sidebar-groups';

/** The name a group made from a menu has until the operator renames it. */
export const NEW_GROUP_LABEL = 'New group';

const MODEL_VERSION = 1;

/** The longest group name, in characters. */
const MAX_LABEL_LENGTH = 40;

/** The built-in groups, in their default order. */
const BUILTIN_GROUPS = Object.freeze([
  Object.freeze({ id: 'workspace', label: 'Workspace' }),
  Object.freeze({ id: 'tools', label: 'Tools' }),
]);

/** Where a panel goes when nothing else places it, and where the panels of a deleted group go. */
const FALLBACK_GROUP = 'tools';

/** The panels that sit in Workspace until the operator moves them. */
const DEFAULT_WORKSPACE = Object.freeze(['artifacts', 'events']);

/** The id of a group the operator added. */
const CUSTOM_ID = /^g-[0-9a-f]{8}$/;

// ---- Model ----

/**
 * Whether `id` names a built-in group, which can be renamed and moved but
 * never deleted.
 * @param {string} id
 * @returns {boolean}
 */
export function isBuiltinGroup(id) {
  return BUILTIN_GROUPS.some((group) => group.id === id);
}

/**
 * A name as a group stores it: trimmed, and cut to 40 characters. The cut
 * counts characters, not UTF-16 units, so it never splits one in two.
 * @param {unknown} raw
 * @returns {string} '' when nothing is left
 */
function cleanLabel(raw) {
  const chars = Array.from(String(raw ?? '').trim());
  return chars.slice(0, MAX_LABEL_LENGTH).join('').trimEnd();
}

/**
 * A placement map without a prototype, so no panel id can read or write an
 * inherited key.
 * @returns {Record<string, string>}
 */
function emptyPlacement() {
  return /** @type {Record<string, string>} */ (Object.create(null));
}

/**
 * The group a panel sits in until the operator places it.
 * @param {string} panelId
 * @returns {string}
 */
function defaultGroupOf(panelId) {
  return DEFAULT_WORKSPACE.includes(panelId) ? 'workspace' : FALLBACK_GROUP;
}

/**
 * The model of a first visit: the two built-in groups, and each named panel
 * placed where the sidebar puts it by default. The Agent row has no
 * placement, since it is always first in the first group.
 * @param {string[]} [panelIds]
 * @returns {GroupsModel}
 */
export function defaultModel(panelIds = []) {
  const placement = emptyPlacement();
  for (const id of panelIds) {
    if (id !== TERMINAL_RAIL_ID) placement[id] = defaultGroupOf(id);
  }
  return {
    version: MODEL_VERSION,
    groups: BUILTIN_GROUPS.map((group) => ({ ...group })),
    placement,
  };
}

/**
 * `value` as a model, or null when it is not one. A model is version 1; its
 * groups have unique, well-formed ids, clean names, and the two built-ins
 * among them; every placement names one of those groups.
 * @param {unknown} value
 * @returns {GroupsModel | null} a fresh copy
 */
function validModel(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const { version, groups, placement } = /** @type {Record<string, unknown>} */ (value);
  if (version !== MODEL_VERSION || !Array.isArray(groups)) return null;
  /** @type {Set<string>} */
  const ids = new Set();
  /** @type {SidebarGroup[]} */
  const copy = [];
  for (const group of groups) {
    if (!group || typeof group !== 'object') return null;
    const { id, label } = /** @type {Record<string, unknown>} */ (group);
    if (typeof id !== 'string' || ids.has(id)) return null;
    if (!isBuiltinGroup(id) && !CUSTOM_ID.test(id)) return null;
    if (typeof label !== 'string' || !label || cleanLabel(label) !== label) return null;
    ids.add(id);
    copy.push({ id, label });
  }
  if (!BUILTIN_GROUPS.every((group) => ids.has(group.id))) return null;
  if (!placement || typeof placement !== 'object' || Array.isArray(placement)) return null;
  const places = emptyPlacement();
  for (const [panelId, groupId] of Object.entries(placement)) {
    if (typeof groupId !== 'string' || !ids.has(groupId)) return null;
    places[panelId] = groupId;
  }
  return { version: MODEL_VERSION, groups: copy, placement: places };
}

/**
 * The model in a stored value: the default when there is none, or when it is
 * not a model.
 * @param {string | null} text
 * @returns {GroupsModel}
 */
function parseModel(text) {
  if (text === null) return defaultModel();
  try {
    return validModel(JSON.parse(text)) ?? defaultModel();
  } catch {
    return defaultModel();
  }
}

/** The model in force, as last read or written. @type {GroupsModel | null} */
let held = null;

/** The stored text `held` was read from or written as, so an unchanged value
 *  is not parsed again. @type {string | null | undefined} */
let heldText;

/** Set once storage refuses a write: from then on, this page's model is the
 *  one in force, and the stored value (now behind it) is no longer read. */
let storageRefused = false;

/** @type {Set<(model: GroupsModel) => void>} */
const listeners = new Set();

/** @returns {string} */
function storageKey() {
  return scopedStorageKey(GROUPS_STORAGE_KEY);
}

/**
 * The model in force: the stored value, parsed again whenever it has changed,
 * or this page's own model when storage is unavailable.
 * @returns {GroupsModel}
 */
function current() {
  if (!storageRefused) {
    try {
      const text = window.localStorage.getItem(storageKey());
      if (!held || text !== heldText) {
        heldText = text;
        held = parseModel(text);
      }
    } catch {
      // Storage is unavailable: the model this page holds stands.
    }
  }
  if (!held) held = defaultModel();
  return held;
}

/**
 * Store `model`, make it the model in force, and tell the listeners.
 * @param {GroupsModel} model
 */
function commit(model) {
  held = model;
  const text = JSON.stringify(model);
  try {
    window.localStorage.setItem(storageKey(), text);
    heldText = text;
  } catch {
    // Storage is unavailable or full. The change lives as long as this page.
    storageRefused = true;
  }
  for (const listener of [...listeners]) listener(loadGroups());
}

/**
 * A copy of the model in force, safe for the caller to change.
 * @returns {GroupsModel}
 */
export function loadGroups() {
  const { groups, placement } = current();
  return {
    version: MODEL_VERSION,
    groups: groups.map((group) => ({ ...group })),
    placement: Object.assign(emptyPlacement(), placement),
  };
}

/**
 * Store a whole model: the persistence step of every change below. A value
 * that is not a model is refused and nothing changes.
 * @param {GroupsModel} model
 * @returns {boolean} whether the model was stored
 */
export function saveGroups(model) {
  const valid = validModel(model);
  if (!valid) return false;
  commit(valid);
  return true;
}

/**
 * The group a panel's row sits in: the first group for the Agent row, the
 * operator's placement for a placed panel, and the default group otherwise.
 * @param {string} panelId
 * @returns {string} a group id of the model in force
 */
export function groupOf(panelId) {
  const { groups, placement } = current();
  if (panelId === TERMINAL_RAIL_ID) return groups[0].id;
  return Object.hasOwn(placement, panelId) ? placement[panelId] : defaultGroupOf(panelId);
}

/**
 * Move a panel's row to another group. The Agent row never moves.
 * @param {string} panelId
 * @param {string} groupId
 * @returns {boolean} whether the row moved
 */
export function moveToGroup(panelId, groupId) {
  if (panelId === TERMINAL_RAIL_ID || groupOf(panelId) === groupId) return false;
  const model = loadGroups();
  if (!model.groups.some((group) => group.id === groupId)) return false;
  model.placement[panelId] = groupId;
  commit(model);
  return true;
}

/**
 * Rename a group. The name is trimmed and cut to 40 characters, and a name
 * that is empty after the trim keeps the old one.
 * @param {string} id
 * @param {string} label
 * @returns {boolean} whether the name changed
 */
export function renameGroup(id, label) {
  const model = loadGroups();
  const group = model.groups.find((g) => g.id === id);
  const clean = cleanLabel(label);
  if (!group || !clean || clean === group.label) return false;
  group.label = clean;
  commit(model);
  return true;
}

/**
 * A group id the model does not use yet.
 * @param {GroupsModel} model
 * @returns {string}
 */
function freshId(model) {
  const taken = new Set(model.groups.map((group) => group.id));
  let id;
  do {
    const bytes = crypto.getRandomValues(new Uint8Array(4));
    id = `g-${Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')}`;
  } while (taken.has(id));
  return id;
}

/**
 * Add an empty group, directly after `afterId`, or last when `afterId` names
 * no group. An empty name becomes {@link NEW_GROUP_LABEL}.
 * @param {string} label
 * @param {string} [afterId]
 * @returns {string} the new group's id
 */
export function addGroup(label, afterId) {
  const model = loadGroups();
  const id = freshId(model);
  const after = afterId === undefined ? -1 : model.groups.findIndex((g) => g.id === afterId);
  const at = after < 0 ? model.groups.length : after + 1;
  model.groups.splice(at, 0, { id, label: cleanLabel(label) || NEW_GROUP_LABEL });
  commit(model);
  return id;
}

/**
 * Delete a group the operator added. Its panels go back to Tools. A built-in
 * group is never deleted.
 * @param {string} id
 * @returns {boolean} whether the group was deleted
 */
export function removeGroup(id) {
  if (isBuiltinGroup(id)) return false;
  const model = loadGroups();
  const at = model.groups.findIndex((group) => group.id === id);
  if (at < 0) return false;
  model.groups.splice(at, 1);
  for (const [panelId, groupId] of Object.entries(model.placement)) {
    if (groupId === id) model.placement[panelId] = FALLBACK_GROUP;
  }
  commit(model);
  return true;
}

/**
 * Move a group one place up or down the sidebar.
 * @param {string} id
 * @param {'up' | 'down'} dir
 * @returns {boolean} whether the group moved (not past either end)
 */
export function reorderGroup(id, dir) {
  const model = loadGroups();
  const at = model.groups.findIndex((group) => group.id === id);
  const to = dir === 'up' ? at - 1 : at + 1;
  if (at < 0 || to < 0 || to >= model.groups.length) return false;
  [model.groups[at], model.groups[to]] = [model.groups[to], model.groups[at]];
  commit(model);
  return true;
}

/**
 * Call `callback` with the new model after every change.
 * @param {(model: GroupsModel) => void} callback
 * @returns {() => void} stops the calls
 */
export function onGroupsChange(callback) {
  listeners.add(callback);
  return () => {
    listeners.delete(callback);
  };
}

// ---- Headings ----

/** The class of a heading's name, which is also the group's accessible name. */
const NAME_CLASS = 'sidebar-group-name';

/** Marks the ⋯ button that opens a group's menu. */
const MENU_BUTTON_ATTR = 'data-group-menu';

/** The class of the field that renames a group in place. */
const RENAME_CLASS = 'sidebar-group-rename';

/** The body-level menu that every group's ⋯ button opens. */
const MENU_ID = 'sidebar-group-menu';

/** ⋯ buttons already wired, so a repaint never adds a second listener.
 *  @type {WeakSet<Element>} */
const wiredMenuButtons = new WeakSet();

/** The ⋯ button the group menu was last opened from. @type {Element | null} */
let menuOpener = null;

/**
 * Give a group element its heading: the `.osprey-section-label` that holds
 * the group's name and its ⋯ button. A heading the page shipped with both
 * parts is adopted as it is; any other heading gets both built. The group is
 * labelled by its name.
 * @param {HTMLElement} groupEl - a `.sidebar-group[data-group]`
 * @param {SidebarGroup} group
 * @param {string} nameId - the id the name takes when it has none yet
 * @returns {HTMLElement} the heading
 */
export function paintGroupHeading(groupEl, group, nameId) {
  let heading = /** @type {HTMLElement | null} */ (
    groupEl.querySelector(':scope > .osprey-section-label')
  );
  if (!heading) {
    heading = document.createElement('div');
    heading.className = 'osprey-section-label';
    groupEl.prepend(heading);
  }
  heading.classList.add('sidebar-group-head');
  let name = /** @type {HTMLElement | null} */ (heading.querySelector(`.${NAME_CLASS}`));
  let button = /** @type {HTMLButtonElement | null} */ (
    heading.querySelector(`[${MENU_BUTTON_ATTR}]`)
  );
  if (!name || !button) {
    name = document.createElement('span');
    name.className = NAME_CLASS;
    button = buildMenuButton();
    heading.replaceChildren(name, button);
  }
  if (!name.id) name.id = nameId;
  name.textContent = group.label;
  groupEl.setAttribute('aria-labelledby', name.id);
  button.setAttribute('aria-label', `${group.label} group options`);
  button.setAttribute(
    'data-tip-detail',
    isBuiltinGroup(group.id) ? 'Rename or move this group' : 'Rename, move or delete this group'
  );
  if (!wiredMenuButtons.has(button)) {
    wiredMenuButtons.add(button);
    const opener = button;
    opener.addEventListener('click', () => toggleGroupMenu(opener));
  }
  return heading;
}

/** @returns {HTMLButtonElement} a group's ⋯ button, unlabelled */
function buildMenuButton() {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'osprey-mini';
  button.setAttribute(MENU_BUTTON_ATTR, '');
  button.setAttribute('aria-haspopup', 'menu');
  button.setAttribute('aria-expanded', 'false');
  button.append(icon('more'));
  return button;
}

/** @returns {HTMLElement} the group menu, made on first use */
function groupMenu() {
  let menu = document.getElementById(MENU_ID);
  if (!menu) {
    menu = document.createElement('div');
    menu.id = MENU_ID;
    menu.className = 'osprey-popover sidebar-group-menu';
    menu.setAttribute('role', 'menu');
    menu.hidden = true;
    document.body.append(menu);
  }
  return menu;
}

/**
 * A group element of the rail, by id.
 * @param {HTMLElement} railEl
 * @param {string} groupId
 * @returns {HTMLElement | null}
 */
function groupElement(railEl, groupId) {
  return /** @type {HTMLElement | null} */ (
    railEl.querySelector(`:scope > .sidebar-group[data-group="${CSS.escape(groupId)}"]`)
  );
}

/**
 * Put focus on a group's ⋯ button: after the menu changed that group, the
 * keyboard continues from the group it was working on.
 * @param {HTMLElement} railEl
 * @param {string} groupId
 */
function focusMenuButton(railEl, groupId) {
  /** @type {HTMLElement | null | undefined} */ (
    groupElement(railEl, groupId)?.querySelector(`[${MENU_BUTTON_ATTR}]`)
  )?.focus();
}

/**
 * Open the menu of the group that `button` heads, or close the menu when this
 * button opened it. A menu open from another group's button moves here.
 * @param {HTMLButtonElement} button
 */
function toggleGroupMenu(button) {
  const menu = groupMenu();
  if (!menu.hidden && menuOpener === button) {
    closePopover(menu);
    return;
  }
  const groupEl = /** @type {HTMLElement | null} */ (button.closest('.sidebar-group[data-group]'));
  const railEl = groupEl?.parentElement;
  const id = groupEl?.getAttribute('data-group');
  if (!railEl || !id) return;
  const { groups } = loadGroups();
  const at = groups.findIndex((group) => group.id === id);
  if (at < 0) return;

  /** @type {HTMLElement[]} */
  const rows = [
    menuItem('Rename', () => {
      startGroupRename(railEl, id);
    }),
    menuItem('New group below', () => {
      startGroupRename(railEl, addGroup(NEW_GROUP_LABEL, id));
    }),
    menuItem(
      'Move up',
      () => {
        reorderGroup(id, 'up');
        focusMenuButton(railEl, id);
      },
      { disabled: at === 0 }
    ),
    menuItem(
      'Move down',
      () => {
        reorderGroup(id, 'down');
        focusMenuButton(railEl, id);
      },
      { disabled: at === groups.length - 1 }
    ),
  ];
  if (!isBuiltinGroup(id)) {
    const sep = document.createElement('div');
    sep.className = 'osprey-popover-sep';
    sep.setAttribute('role', 'separator');
    rows.push(
      sep,
      menuItem(
        'Delete group',
        () => {
          removeGroup(id);
          focusMenuButton(railEl, FALLBACK_GROUP);
        },
        { danger: true }
      )
    );
  }
  menu.setAttribute('aria-label', `${groups[at].label} group`);
  menu.replaceChildren(...rows);
  menuOpener = button;
  openPopover(menu, button, {
    place: 'beside-right',
    edge: button.closest('#sidebar') ?? railEl,
  });
}

/**
 * One row of the group menu. The menu closes before the row's action runs,
 * so an action that moves focus (the rename field) keeps it.
 * @param {string} text
 * @param {() => void} run
 * @param {{disabled?: boolean, danger?: boolean}} [flags]
 * @returns {HTMLButtonElement}
 */
function menuItem(text, run, { disabled = false, danger = false } = {}) {
  const item = document.createElement('button');
  item.type = 'button';
  item.className = 'osprey-popover-item';
  item.classList.toggle('danger', danger);
  item.setAttribute('role', 'menuitem');
  item.textContent = text;
  item.disabled = disabled;
  item.addEventListener('click', () => {
    closePopover(groupMenu());
    run();
  });
  return item;
}

/**
 * Swap a group's name for a text field that holds it, and focus the field.
 * Enter, or leaving the field, renames the group; Escape keeps the old name,
 * and so does an empty field. After Enter or Escape, focus goes to the
 * group's ⋯ button, where the keyboard was working; leaving the field with
 * the pointer moves focus to wherever the pointer went.
 *
 * A collapsed sidebar shows no names, so it opens no field: a group made
 * there keeps its default name until the operator renames it.
 * @param {HTMLElement} railEl - the rail the group's element is in
 * @param {string} groupId
 * @returns {boolean} whether the field opened
 */
export function startGroupRename(railEl, groupId) {
  if (document.documentElement.getAttribute('data-sidebar') === 'collapsed') return false;
  const groupEl = groupElement(railEl, groupId);
  const name = /** @type {HTMLElement | null | undefined} */ (
    groupEl?.querySelector(`.${NAME_CLASS}`)
  );
  if (!groupEl || !name || groupEl.querySelector(`.${RENAME_CLASS}`)) return false;

  const field = document.createElement('input');
  field.type = 'text';
  field.className = RENAME_CLASS;
  field.value = name.textContent ?? '';
  field.maxLength = MAX_LABEL_LENGTH;
  field.setAttribute('aria-label', 'Group name');
  name.hidden = true;
  name.after(field);
  field.focus();
  field.select();

  let finished = false;
  /** @param {boolean} keep @param {boolean} refocus */
  const finish = (keep, refocus) => {
    // Removing the focused field blurs it, which would finish a second time.
    if (finished) return;
    finished = true;
    const value = field.value;
    field.remove();
    name.hidden = false;
    if (keep) renameGroup(groupId, value);
    if (refocus) focusMenuButton(railEl, groupId);
  };
  field.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' && event.key !== 'Escape') return;
    // The field answers these keys itself: the page's own Escape and Enter
    // handlers must not act on them as well.
    event.preventDefault();
    event.stopPropagation();
    finish(event.key === 'Enter', true);
  });
  field.addEventListener('blur', () => finish(true, false));
  return true;
}
