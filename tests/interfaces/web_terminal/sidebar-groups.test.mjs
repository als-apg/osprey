// @ts-check
/**
 * The sidebar's groups (sidebar-groups.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-groups.test.mjs
 *
 * The model: the per-user groups kept in this browser under the scoped
 * `osprey-sidebar-groups` key — the defaults and the default placement,
 * rename, add, delete, reorder, moving a panel, the stored round trip, and
 * the reset of a stored value that is not a model. The headings: the ⋯ menu
 * each heading opens through the real popover.js, its rows and what they do,
 * and the rename field.
 *
 * The heading tests draw a real rail (panel-rail.js createRail), because the
 * menu acts on the group elements the rail renders and the rail moves its
 * rows whenever the model changes. Nothing is mocked. The model is re-read
 * from storage on every access, so clearing storage before each test is
 * what isolates them; the one test that needs storage to refuse a write
 * imports a fresh copy of the module, so the refusal stays with that copy.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { byId, qs } from '../_support/dom.mjs';
import {
  GROUPS_STORAGE_KEY,
  NEW_GROUP_LABEL,
  addGroup,
  defaultModel,
  groupOf,
  isBuiltinGroup,
  loadGroups,
  moveToGroup,
  onGroupsChange,
  removeGroup,
  renameGroup,
  reorderGroup,
  saveGroups,
  startGroupRename,
} from '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-groups.js';
import {
  createRail,
  entryIds,
} from '../../../src/osprey/interfaces/web_terminal/static/js/panel-rail.js';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const SCOPE_ATTR = 'data-osprey-storage-scope';

/** The group ids of the model in force, in sidebar order. */
const groupIds = () => loadGroups().groups.map((group) => group.id);

/** The group labels of the model in force, in sidebar order. */
const groupLabels = () => loadGroups().groups.map((group) => group.label);

/** @param {string} id @returns {string | undefined} the label of group `id` */
const labelOf = (id) => loadGroups().groups.find((group) => group.id === id)?.label;

/** The stored value under `key`, parsed. @param {string} [key] */
const stored = (key = GROUPS_STORAGE_KEY) => JSON.parse(localStorage.getItem(key) ?? 'null');

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute(SCOPE_ATTR);
  document.documentElement.removeAttribute('data-sidebar');
  document.body.innerHTML = '';
});

afterEach(() => {
  document.documentElement.removeAttribute(SCOPE_ATTR);
  vi.restoreAllMocks();
});

/* ---- the model ----------------------------------------------------------- */

describe('the default model', () => {
  test('two built-in groups, and nothing placed', () => {
    expect(defaultModel()).toEqual({
      version: 1,
      groups: [
        { id: 'workspace', label: 'Workspace' },
        { id: 'tools', label: 'Tools' },
      ],
      placement: {},
    });
  });

  test('Artifacts and Events sit in Workspace, every other panel in Tools', () => {
    const model = defaultModel(['terminal', 'artifacts', 'events', 'ariel', 'lattice']);
    expect({ ...model.placement }).toEqual({
      artifacts: 'workspace',
      events: 'workspace',
      ariel: 'tools',
      lattice: 'tools',
    });
  });

  test('a first visit reads the default model, and places panels as the sidebar always has', () => {
    expect(loadGroups()).toEqual(defaultModel());
    expect(groupOf('artifacts')).toBe('workspace');
    expect(groupOf('events')).toBe('workspace');
    expect(groupOf('ariel')).toBe('tools');
    expect(groupOf('a-custom-panel')).toBe('tools');
  });

  test('the Agent row is in the first group, whichever group that is', () => {
    expect(groupOf('terminal')).toBe('workspace');
    reorderGroup('tools', 'up');
    expect(groupOf('terminal')).toBe('tools');
  });

  test('the built-in groups are workspace and tools', () => {
    expect(isBuiltinGroup('workspace')).toBe(true);
    expect(isBuiltinGroup('tools')).toBe(true);
    expect(isBuiltinGroup(addGroup('Mine'))).toBe(false);
  });
});

describe('rename', () => {
  test('a name is trimmed', () => {
    expect(renameGroup('tools', '  Instruments  ')).toBe(true);
    expect(labelOf('tools')).toBe('Instruments');
  });

  test('an empty name keeps the old one', () => {
    for (const blank of ['', '   ']) {
      expect(renameGroup('workspace', blank)).toBe(false);
      expect(labelOf('workspace')).toBe('Workspace');
    }
  });

  test('a long name is cut to 40 characters', () => {
    renameGroup('tools', 'x'.repeat(50));
    expect(labelOf('tools')).toBe('x'.repeat(40));
  });

  test('the cut counts characters, so it never splits one', () => {
    renameGroup('tools', '🛰'.repeat(45));
    expect(Array.from(labelOf('tools') ?? '')).toEqual(Array(40).fill('🛰'));
  });

  test('the same name, or an unknown group, changes nothing', () => {
    const seen = vi.fn();
    const stop = onGroupsChange(seen);
    expect(renameGroup('tools', 'Tools')).toBe(false);
    expect(renameGroup('nope', 'Anything')).toBe(false);
    stop();
    expect(seen).not.toHaveBeenCalled();
  });
});

describe('add, delete and reorder', () => {
  test('a group added after another sits directly below it, with a fresh id', () => {
    const id = addGroup('Beamline', 'workspace');
    expect(id).toMatch(/^g-[0-9a-f]{8}$/);
    expect(groupIds()).toEqual(['workspace', id, 'tools']);
    expect(labelOf(id)).toBe('Beamline');
  });

  test('a group added after no group goes last, and an empty name becomes the default', () => {
    const id = addGroup('   ');
    expect(groupIds()).toEqual(['workspace', 'tools', id]);
    expect(labelOf(id)).toBe(NEW_GROUP_LABEL);
  });

  test('every added group gets its own id', () => {
    const ids = new Set(Array.from({ length: 20 }, () => addGroup('Same')));
    expect(ids.size).toBe(20);
  });

  test('a built-in group is never deleted', () => {
    expect(removeGroup('workspace')).toBe(false);
    expect(removeGroup('tools')).toBe(false);
    expect(groupIds()).toEqual(['workspace', 'tools']);
  });

  test('deleting a group sends its panels back to Tools, Artifacts included', () => {
    const id = addGroup('Beamline');
    moveToGroup('ariel', id);
    moveToGroup('artifacts', id);
    moveToGroup('lattice', 'workspace');

    expect(removeGroup(id)).toBe(true);

    expect(groupIds()).toEqual(['workspace', 'tools']);
    expect(groupOf('ariel')).toBe('tools');
    expect(groupOf('artifacts')).toBe('tools');
    expect(groupOf('lattice')).toBe('workspace');
  });

  test('deleting an unknown group changes nothing', () => {
    expect(removeGroup('g-00000000')).toBe(false);
  });

  test('a group moves one place up or down, built-in ones too, never past an end', () => {
    const id = addGroup('Beamline', 'workspace');
    expect(reorderGroup(id, 'up')).toBe(true);
    expect(groupIds()).toEqual([id, 'workspace', 'tools']);
    expect(reorderGroup(id, 'up')).toBe(false);
    expect(reorderGroup('tools', 'down')).toBe(false);
    expect(reorderGroup('workspace', 'down')).toBe(true);
    expect(groupIds()).toEqual([id, 'tools', 'workspace']);
    expect(reorderGroup('nope', 'up')).toBe(false);
  });
});

describe('moving a panel', () => {
  test('a panel moves to another group', () => {
    expect(moveToGroup('ariel', 'workspace')).toBe(true);
    expect(groupOf('ariel')).toBe('workspace');
  });

  test('a page-owned row moves like any other', () => {
    expect(moveToGroup('artifacts', 'tools')).toBe(true);
    expect(groupOf('artifacts')).toBe('tools');
  });

  test('the Agent row never moves', () => {
    expect(moveToGroup('terminal', 'tools')).toBe(false);
    expect(groupOf('terminal')).toBe('workspace');
  });

  test('a move to an unknown group, or to the group it is in, changes nothing', () => {
    expect(moveToGroup('ariel', 'nope')).toBe(false);
    expect(moveToGroup('ariel', 'tools')).toBe(false);
    expect(localStorage.getItem(GROUPS_STORAGE_KEY)).toBeNull();
  });
});

describe('persistence', () => {
  test('every change is stored, and a reload reads it back', async () => {
    const id = addGroup('Beamline', 'workspace');
    renameGroup('tools', 'Instruments');
    moveToGroup('ariel', id);

    expect(stored()).toEqual({
      version: 1,
      groups: [
        { id: 'workspace', label: 'Workspace' },
        { id, label: 'Beamline' },
        { id: 'tools', label: 'Instruments' },
      ],
      placement: { ariel: id },
    });

    vi.resetModules();
    const reloaded = await import(`${JS}/sidebar-groups.js`);
    expect(reloaded.loadGroups()).toEqual(loadGroups());
    expect(reloaded.groupOf('ariel')).toBe(id);
  });

  test('the groups are per user on a shared browser', () => {
    document.documentElement.setAttribute(SCOPE_ATTR, 'alice');
    renameGroup('tools', 'Alice tools');
    expect(stored(`${GROUPS_STORAGE_KEY}--alice`).groups[1].label).toBe('Alice tools');
    expect(localStorage.getItem(GROUPS_STORAGE_KEY)).toBeNull();

    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
    expect(labelOf('tools')).toBe('Tools');
  });

  test('a value another tab stored is the base of the next change here', () => {
    renameGroup('tools', 'Mine');
    const other = defaultModel();
    other.groups[0].label = 'Theirs';
    localStorage.setItem(GROUPS_STORAGE_KEY, JSON.stringify(other));

    renameGroup('tools', 'Instruments');

    expect(groupLabels()).toEqual(['Theirs', 'Instruments']);
  });

  const valid = defaultModel();
  test.each([
    ['not JSON', 'not json'],
    ['not an object', '[]'],
    ['null', 'null'],
    ['another version', JSON.stringify({ ...valid, version: 2 })],
    ['groups that are not a list', JSON.stringify({ ...valid, groups: {} })],
    ['a built-in group missing', JSON.stringify({ ...valid, groups: [valid.groups[0]] })],
    [
      'a repeated id',
      JSON.stringify({ ...valid, groups: [...valid.groups, { id: 'tools', label: 'Again' }] }),
    ],
    [
      'a malformed id',
      JSON.stringify({ ...valid, groups: [...valid.groups, { id: 'g-xyz', label: 'Bad' }] }),
    ],
    [
      'an empty name',
      JSON.stringify({ ...valid, groups: [valid.groups[0], { id: 'tools', label: '' }] }),
    ],
    [
      'an untrimmed name',
      JSON.stringify({ ...valid, groups: [valid.groups[0], { id: 'tools', label: ' Tools' }] }),
    ],
    [
      'a name over 40 characters',
      JSON.stringify({
        ...valid,
        groups: [valid.groups[0], { id: 'tools', label: 'x'.repeat(41) }],
      }),
    ],
    ['a placement that is not an object', JSON.stringify({ ...valid, placement: [] })],
    ['a placement naming no group', JSON.stringify({ ...valid, placement: { ariel: 'g-0000' } })],
  ])('a stored value with %s reads as the default', (_case, raw) => {
    localStorage.setItem(GROUPS_STORAGE_KEY, raw);

    expect(loadGroups()).toEqual(defaultModel());
    expect(groupOf('ariel')).toBe('tools');

    // The next change overwrites it with a model.
    renameGroup('tools', 'Instruments');
    expect(stored().groups[1].label).toBe('Instruments');
  });

  test('saveGroups stores a model and refuses anything else', () => {
    const model = defaultModel();
    model.groups[1].label = 'Instruments';
    expect(saveGroups(model)).toBe(true);
    expect(labelOf('tools')).toBe('Instruments');

    const broken = defaultModel();
    broken.placement.ariel = 'nope';
    expect(saveGroups(broken)).toBe(false);
    expect(labelOf('tools')).toBe('Instruments');
  });

  test('storage that refuses a write keeps the change for this page', async () => {
    const real = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
    // Safari's private mode: reads answer, writes throw.
    const refusing = {
      getItem: () => null,
      setItem() {
        throw new Error('QuotaExceededError');
      },
    };
    Object.defineProperty(globalThis, 'localStorage', {
      value: refusing,
      configurable: true,
      writable: true,
    });
    try {
      vi.resetModules();
      const fresh = await import(`${JS}/sidebar-groups.js`);
      expect(fresh.renameGroup('tools', 'Instruments')).toBe(true);
      expect(fresh.moveToGroup('ariel', 'workspace')).toBe(true);
      expect(fresh.loadGroups().groups[1].label).toBe('Instruments');
      expect(fresh.groupOf('ariel')).toBe('workspace');
    } finally {
      if (real) Object.defineProperty(globalThis, 'localStorage', real);
    }
  });
});

describe('onGroupsChange', () => {
  test('a listener hears every change, with the new model, until it stops', () => {
    /** @type {string[][]} */
    const seen = [];
    const stop = onGroupsChange((model) => seen.push(model.groups.map((g) => g.label)));

    renameGroup('tools', 'Instruments');
    addGroup('Beamline');
    stop();
    renameGroup('tools', 'Tools');

    expect(seen).toEqual([
      ['Workspace', 'Instruments'],
      ['Workspace', 'Instruments', 'Beamline'],
    ]);
  });
});

/* ---- the headings -------------------------------------------------------- */

const PANELS = [
  { id: 'terminal', label: 'Agent' },
  { id: 'artifacts', label: 'Artifacts' },
  { id: 'ariel', label: 'Ariel' },
  { id: 'lattice', label: 'Lattice' },
];

/**
 * A rail drawn inside the sidebar, as the page has it.
 * @returns {HTMLElement}
 */
function mountRail() {
  document.body.innerHTML = '<nav class="sidebar" id="sidebar"><div id="panel-rail"></div></nav>';
  const rail = byId('panel-rail');
  createRail(rail, PANELS);
  return rail;
}

/** @param {HTMLElement} rail @param {string} id @returns {HTMLElement} */
const groupEl = (rail, id) => qs(rail, `:scope > .sidebar-group[data-group="${id}"]`);

/** @param {HTMLElement} rail @param {string} id @returns {HTMLButtonElement} */
const menuButton = (rail, id) => qs(groupEl(rail, id), '[data-group-menu]', HTMLButtonElement);

/** @returns {HTMLElement | null} the group menu, when it has been made */
const menu = () => document.getElementById('sidebar-group-menu');

/** @returns {HTMLButtonElement[]} the rows of the open menu */
function menuItems() {
  const rows = menu()?.querySelectorAll('[role="menuitem"]') ?? [];
  return /** @type {HTMLButtonElement[]} */ ([...rows]);
}

/** The rows of the open menu, as `label`, `label (disabled)` or `label (danger)`. */
function menuRows() {
  return menuItems().map((row) => {
    if (row.disabled) return `${row.textContent} (disabled)`;
    return row.classList.contains('danger') ? `${row.textContent} (danger)` : row.textContent;
  });
}

/** @param {string} label @returns {HTMLButtonElement} the open menu's row reading `label` */
function menuRow(label) {
  const row = menuItems().find((el) => el.textContent === label);
  if (!row) throw new Error(`no menu row ${label}`);
  return row;
}

/** @param {HTMLElement} rail @param {string} id @returns {HTMLInputElement | null} */
function renameField(rail, id) {
  const field = groupEl(rail, id).querySelector('.sidebar-group-rename');
  return field instanceof HTMLInputElement ? field : null;
}

/** @param {HTMLElement} el @param {string} key */
function press(el, key) {
  el.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
}

afterEach(() => {
  // Close whatever a test left open: the dismissal listeners are popover.js's.
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
});

describe('group headings', () => {
  test('each heading holds the group name and a ⋯ button, and names its group', () => {
    const rail = mountRail();
    for (const [id, label] of [
      ['workspace', 'Workspace'],
      ['tools', 'Tools'],
    ]) {
      const group = groupEl(rail, id);
      const heading = qs(group, ':scope > .osprey-section-label');
      expect(heading).toBe(group.firstElementChild);
      const name = qs(heading, '.sidebar-group-name');
      expect(name.textContent).toBe(label);
      expect(group.getAttribute('aria-labelledby')).toBe(name.id);
      const button = menuButton(rail, id);
      expect(button.classList.contains('osprey-mini')).toBe(true);
      expect(button.getAttribute('aria-label')).toBe(`${label} group options`);
      expect(button.getAttribute('aria-haspopup')).toBe('menu');
    }
  });

  test('a renamed group shows its new name at once', () => {
    const rail = mountRail();
    renameGroup('tools', 'Instruments');
    expect(qs(groupEl(rail, 'tools'), '.sidebar-group-name').textContent).toBe('Instruments');
    expect(menuButton(rail, 'tools').getAttribute('aria-label')).toBe('Instruments group options');
  });
});

describe('the group menu', () => {
  test('⋯ opens a body-level menu: a built-in group has no Delete', () => {
    const rail = mountRail();
    menuButton(rail, 'workspace').click();

    const opened = /** @type {HTMLElement} */ (menu());
    expect(opened.hidden).toBe(false);
    expect(opened.parentElement).toBe(document.body);
    expect(opened.classList.contains('osprey-popover')).toBe(true);
    expect(opened.getAttribute('role')).toBe('menu');
    expect(opened.getAttribute('aria-label')).toBe('Workspace group');
    expect(menuButton(rail, 'workspace').getAttribute('aria-expanded')).toBe('true');
    expect(menuRows()).toEqual(['Rename', 'New group below', 'Move up (disabled)', 'Move down']);
  });

  test('the last group cannot move down, and a group the operator added can be deleted', () => {
    const rail = mountRail();
    const id = addGroup('Beamline');
    menuButton(rail, id).click();
    expect(menuRows()).toEqual([
      'Rename',
      'New group below',
      'Move up',
      'Move down (disabled)',
      'Delete group (danger)',
    ]);
  });

  test('a second click closes the menu; another group’s ⋯ moves it there', () => {
    const rail = mountRail();
    menuButton(rail, 'workspace').click();
    menuButton(rail, 'workspace').click();
    expect(menu()?.hidden).toBe(true);

    menuButton(rail, 'workspace').click();
    menuButton(rail, 'tools').click();
    expect(menu()?.hidden).toBe(false);
    expect(menu()?.getAttribute('aria-label')).toBe('Tools group');
    expect(menuButton(rail, 'workspace').getAttribute('aria-expanded')).toBe('false');
  });

  test('Move down moves the group, closes the menu and keeps focus on its ⋯', () => {
    const rail = mountRail();
    menuButton(rail, 'workspace').click();
    menuRow('Move down').click();

    expect(menu()?.hidden).toBe(true);
    expect(groupIds()).toEqual(['tools', 'workspace']);
    const order = [...rail.children].map((el) => el.getAttribute('data-group'));
    expect(order).toEqual(['tools', 'workspace']);
    expect(document.activeElement).toBe(menuButton(rail, 'workspace'));
  });

  test('Delete group removes it, and its rows go back to Tools', () => {
    const rail = mountRail();
    const id = addGroup('Beamline');
    moveToGroup('ariel', id);
    menuButton(rail, id).click();
    menuRow('Delete group').click();

    expect(groupIds()).toEqual(['workspace', 'tools']);
    expect(rail.querySelector(`[data-group="${id}"]`)).toBeNull();
    const ariel = rail.querySelector('[data-panel-id="ariel"]');
    expect(groupEl(rail, 'tools').contains(ariel)).toBe(true);
    expect(document.activeElement).toBe(menuButton(rail, 'tools'));
  });

  test('New group below adds an empty group under this one and opens its name', () => {
    const rail = mountRail();
    menuButton(rail, 'workspace').click();
    menuRow('New group below').click();

    const ids = groupIds();
    expect(ids).toHaveLength(3);
    const id = ids[1];
    expect(labelOf(id)).toBe(NEW_GROUP_LABEL);
    const field = renameField(rail, id);
    expect(field).not.toBeNull();
    expect(document.activeElement).toBe(field);
    expect(field?.value).toBe(NEW_GROUP_LABEL);
  });
});

describe('rename in place', () => {
  test('Rename swaps the name for a field holding it, focused', () => {
    const rail = mountRail();
    menuButton(rail, 'tools').click();
    menuRow('Rename').click();

    const field = renameField(rail, 'tools');
    const name = qs(groupEl(rail, 'tools'), '.sidebar-group-name');
    expect(field?.value).toBe('Tools');
    expect(field?.maxLength).toBe(40);
    expect(field?.getAttribute('aria-label')).toBe('Group name');
    expect(name.hidden).toBe(true);
    expect(document.activeElement).toBe(field);
  });

  test('Enter keeps the trimmed name and puts focus back on the ⋯ button', () => {
    const rail = mountRail();
    expect(startGroupRename(rail, 'tools')).toBe(true);
    const field = /** @type {HTMLInputElement} */ (renameField(rail, 'tools'));
    field.value = '  Instruments ';
    press(field, 'Enter');

    expect(labelOf('tools')).toBe('Instruments');
    expect(renameField(rail, 'tools')).toBeNull();
    const name = qs(groupEl(rail, 'tools'), '.sidebar-group-name');
    expect(name.hidden).toBe(false);
    expect(name.textContent).toBe('Instruments');
    expect(document.activeElement).toBe(menuButton(rail, 'tools'));
  });

  test('Escape keeps the old name', () => {
    const rail = mountRail();
    startGroupRename(rail, 'tools');
    const field = /** @type {HTMLInputElement} */ (renameField(rail, 'tools'));
    field.value = 'Instruments';
    press(field, 'Escape');

    expect(labelOf('tools')).toBe('Tools');
    expect(renameField(rail, 'tools')).toBeNull();
    expect(document.activeElement).toBe(menuButton(rail, 'tools'));
  });

  test('leaving the field keeps the new name', () => {
    const rail = mountRail();
    startGroupRename(rail, 'workspace');
    const field = /** @type {HTMLInputElement} */ (renameField(rail, 'workspace'));
    field.value = 'Operations';
    field.dispatchEvent(new FocusEvent('blur'));

    expect(labelOf('workspace')).toBe('Operations');
    expect(renameField(rail, 'workspace')).toBeNull();
  });

  test('an empty field keeps the old name', () => {
    const rail = mountRail();
    startGroupRename(rail, 'tools');
    const field = /** @type {HTMLInputElement} */ (renameField(rail, 'tools'));
    field.value = '   ';
    press(field, 'Enter');

    expect(labelOf('tools')).toBe('Tools');
    expect(qs(groupEl(rail, 'tools'), '.sidebar-group-name').textContent).toBe('Tools');
  });

  test('one field at a time per group, and none for an unknown group', () => {
    const rail = mountRail();
    expect(startGroupRename(rail, 'tools')).toBe(true);
    expect(startGroupRename(rail, 'tools')).toBe(false);
    expect(startGroupRename(rail, 'nope')).toBe(false);
  });

  test('the collapsed sidebar shows no names, so it opens no field', () => {
    const rail = mountRail();
    document.documentElement.setAttribute('data-sidebar', 'collapsed');
    expect(startGroupRename(rail, 'tools')).toBe(false);
    expect(renameField(rail, 'tools')).toBeNull();
  });

  test('renaming moves no row', () => {
    const rail = mountRail();
    const before = entryIds(rail);
    startGroupRename(rail, 'tools');
    press(/** @type {HTMLInputElement} */ (renameField(rail, 'tools')), 'Enter');
    expect(entryIds(rail)).toEqual(before);
  });
});
