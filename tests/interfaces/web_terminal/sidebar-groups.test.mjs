// @ts-check
/**
 * The sidebar's groups (sidebar-groups.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-groups.test.mjs
 *
 * The model: the per-user groups kept in this browser under the scoped
 * `osprey-sidebar-groups` key — the defaults and the default placement,
 * rename, add, delete, reorder, moving a panel, the stored round trip, and
 * the reset of a stored value that is not a model. The headings: the menu a
 * right-click (or Shift+F10 / the menu key) on a heading opens through the
 * real shared context menu, its rows and what they do, and the rename field.
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
  HEADLESS_GROUP,
  NEW_GROUP_LABEL,
  addGroup,
  defaultModel,
  groupOf,
  isBuiltinGroup,
  isHeadlessGroup,
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
  test('three built-in groups, and nothing placed', () => {
    expect(defaultModel()).toEqual({
      version: 3,
      groups: [
        { id: 'workspace', label: 'Sessions' },
        { id: 'tools', label: 'Apps' },
        { id: 'system', label: 'System' },
      ],
      placement: {},
    });
  });

  test('Artifacts sit in Workspace, Events and Health in System, every other panel in Apps', () => {
    const model = defaultModel([
      'terminal',
      'artifacts',
      'events',
      'system-health',
      'ariel',
      'lattice',
    ]);
    expect({ ...model.placement }).toEqual({
      artifacts: 'workspace',
      events: 'system',
      'system-health': 'system',
      ariel: 'tools',
      lattice: 'tools',
    });
  });

  test('a first visit reads the default model, and places panels as the sidebar always has', () => {
    expect(loadGroups()).toEqual(defaultModel());
    expect(groupOf('artifacts')).toBe('workspace');
    expect(groupOf('events')).toBe('system');
    expect(groupOf('system-health')).toBe('system');
    expect(groupOf('ariel')).toBe('tools');
    expect(groupOf('a-custom-panel')).toBe('tools');
  });

  test('the Agent row is in the headless first group, and no group moves above it', () => {
    expect(groupOf('terminal')).toBe('workspace');
    expect(reorderGroup('tools', 'up')).toBe(false);
    expect(groupOf('terminal')).toBe('workspace');
  });

  test('the built-in groups are workspace, tools and system', () => {
    expect(isBuiltinGroup('workspace')).toBe(true);
    expect(isBuiltinGroup('tools')).toBe(true);
    expect(isBuiltinGroup('system')).toBe(true);
    expect(isBuiltinGroup(addGroup('Mine'))).toBe(false);
  });
});

describe('a value written under the current version', () => {
  test.each([
    ['events', 'workspace'],
    ['system-health', 'tools'],
  ])('%s moved to %s survives a reload', (panel, group) => {
    expect(moveToGroup(panel, group)).toBe(true);
    expect(stored().version).toBe(3);
    expect(groupOf(panel)).toBe(group);
    // A reload reads the stored text again, with nothing held in memory.
    localStorage.setItem(GROUPS_STORAGE_KEY, JSON.stringify(stored(), null, 1));
    expect(groupOf(panel)).toBe(group);
  });
});

describe('a value of an older version', () => {
  const legacy = {
    version: 1,
    groups: [
      { id: 'tools', label: 'Instruments' },
      { id: 'workspace', label: 'Workspace' },
    ],
    placement: { events: 'workspace', ariel: 'tools', 'system-health': 'tools' },
  };
  const custom = { id: 'g-0a1b2c3d', label: "test'" };

  test('an untouched old default gains System holding Events and Health', () => {
    localStorage.setItem(GROUPS_STORAGE_KEY, JSON.stringify(legacy));
    expect(groupIds()).toEqual(['workspace', 'tools', 'system']);
    expect(groupOf('events')).toBe('system');
    expect(groupOf('system-health')).toBe('system');
    expect(groupOf('ariel')).toBe('tools');
  });

  test.each([1, 2])(
    'version %i: Events and Health go to System from whichever group held them',
    (version) => {
      for (const to of ['workspace', 'tools', custom.id]) {
        localStorage.setItem(
          GROUPS_STORAGE_KEY,
          JSON.stringify({
            version,
            groups: [...legacy.groups, { id: 'system', label: 'System' }, custom],
            placement: { events: to, 'system-health': to, bluesky: custom.id },
          })
        );
        expect(groupOf('events')).toBe('system');
        expect(groupOf('system-health')).toBe('system');
      }
    }
  );

  test("the owner's layout: Events in Apps, a custom group holding Bluesky", () => {
    localStorage.setItem(
      GROUPS_STORAGE_KEY,
      JSON.stringify({
        version: 2,
        groups: [
          { id: 'workspace', label: 'Sessions' },
          { id: 'tools', label: 'Apps' },
          custom,
          { id: 'system', label: 'System' },
        ],
        placement: { events: 'tools', bluesky: custom.id },
      })
    );
    expect(groupOf('events')).toBe('system');
    expect(groupOf('bluesky')).toBe(custom.id);
    expect(labelOf(custom.id)).toBe("test'");
    expect(groupIds()).toEqual(['workspace', 'tools', custom.id, 'system']);
  });

  test('custom groups and their other panels survive', () => {
    localStorage.setItem(
      GROUPS_STORAGE_KEY,
      JSON.stringify({
        ...legacy,
        groups: [...legacy.groups, custom],
        placement: { ...legacy.placement, ariel: custom.id },
      })
    );
    expect(groupIds()).toEqual(['workspace', 'tools', custom.id, 'system']);
    expect(groupOf('ariel')).toBe(custom.id);
  });

  test('the next change writes the current version, after which a move of Events sticks', () => {
    localStorage.setItem(GROUPS_STORAGE_KEY, JSON.stringify({ ...legacy, version: 2 }));
    expect(groupOf('events')).toBe('system');
    expect(moveToGroup('events', 'tools')).toBe(true);
    expect(stored().version).toBe(3);
    // A reload reads the stored text again, with nothing held in memory.
    localStorage.setItem(GROUPS_STORAGE_KEY, JSON.stringify(stored(), null, 1));
    expect(groupOf('events')).toBe('tools');
  });

  test('moving a panel back to its default stores no placement', () => {
    moveToGroup('events', 'tools');
    expect(stored().placement.events).toBe('tools');
    moveToGroup('events', 'system');
    expect(stored().placement).not.toHaveProperty('events');
  });

  test('keeps the arrangement below Sessions and gains System as the last group', () => {
    localStorage.setItem(GROUPS_STORAGE_KEY, JSON.stringify(legacy));
    expect(groupIds()).toEqual(['workspace', 'tools', 'system']);
    expect(labelOf('workspace')).toBe('Sessions');
    expect(labelOf('tools')).toBe('Instruments');
    expect(labelOf('system')).toBe('System');
    expect(groupOf('ariel')).toBe('tools');
  });

  test('a stored default with no placements puts Events in System', () => {
    localStorage.setItem(GROUPS_STORAGE_KEY, JSON.stringify({ ...legacy, placement: {} }));
    expect(groupOf('events')).toBe('system');
  });
});

describe('rename', () => {
  test('a name is trimmed', () => {
    expect(renameGroup('tools', '  Instruments  ')).toBe(true);
    expect(labelOf('tools')).toBe('Instruments');
  });

  test('an empty name keeps the old one', () => {
    for (const blank of ['', '   ']) {
      expect(renameGroup('tools', blank)).toBe(false);
      expect(labelOf('tools')).toBe('Apps');
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

  test('a saved layout holding the old default name reads as Apps, and keeps a chosen name', () => {
    /** @param {string} label */
    const saved = (label) =>
      JSON.stringify({
        version: 3,
        groups: [
          { id: 'workspace', label: 'Sessions' },
          { id: 'tools', label },
          { id: 'system', label: 'System' },
        ],
        placement: { lattice: 'tools' },
      });
    localStorage.setItem(GROUPS_STORAGE_KEY, saved('Tools'));
    expect(labelOf('tools')).toBe('Apps');
    localStorage.setItem(GROUPS_STORAGE_KEY, saved('Instruments'));
    expect(labelOf('tools')).toBe('Instruments');
  });

  test('the same name, or an unknown group, changes nothing', () => {
    const seen = vi.fn();
    const stop = onGroupsChange(seen);
    expect(renameGroup('tools', 'Apps')).toBe(false);
    expect(renameGroup('nope', 'Anything')).toBe(false);
    stop();
    expect(seen).not.toHaveBeenCalled();
  });
});

describe('the headless Sessions group', () => {
  test('is the workspace group, and only it', () => {
    expect(HEADLESS_GROUP).toBe('workspace');
    expect(isHeadlessGroup('workspace')).toBe(true);
    expect(isHeadlessGroup('tools')).toBe(false);
    expect(isHeadlessGroup(addGroup('Mine'))).toBe(false);
  });

  test('is never renamed, moved, or deleted', () => {
    const seen = vi.fn();
    const stop = onGroupsChange(seen);
    expect(renameGroup('workspace', 'Operations')).toBe(false);
    expect(reorderGroup('workspace', 'down')).toBe(false);
    expect(reorderGroup('tools', 'up')).toBe(false);
    expect(removeGroup('workspace')).toBe(false);
    stop();
    expect(seen).not.toHaveBeenCalled();
    expect(groupIds()).toEqual(['workspace', 'tools', 'system']);
    expect(labelOf('workspace')).toBe('Sessions');
  });

  test('takes panels like any group, and is what a move menu calls Sessions', () => {
    expect(moveToGroup('ariel', 'workspace')).toBe(true);
    expect(groupOf('ariel')).toBe('workspace');
    expect(groupLabels()[0]).toBe('Sessions');
  });

  test('a current-version value that renamed or lowered it reads with its name and place back', () => {
    const custom = { id: 'g-0a1b2c3d', label: 'Beamline' };
    localStorage.setItem(
      GROUPS_STORAGE_KEY,
      JSON.stringify({
        version: 3,
        groups: [
          { id: 'tools', label: 'Instruments' },
          custom,
          { id: 'workspace', label: 'Operations' },
          { id: 'system', label: 'System' },
        ],
        placement: { ariel: 'workspace', bluesky: custom.id },
      })
    );
    expect(groupIds()).toEqual(['workspace', 'tools', custom.id, 'system']);
    expect(groupLabels()).toEqual(['Sessions', 'Instruments', 'Beamline', 'System']);
    expect(groupOf('ariel')).toBe('workspace');
    expect(groupOf('bluesky')).toBe(custom.id);
    expect(groupOf('terminal')).toBe('workspace');
  });

  test('a custom group may still be called Workspace: names were never unique', () => {
    const id = addGroup('Workspace');
    expect(labelOf(id)).toBe('Workspace');
  });
});

describe('add, delete and reorder', () => {
  test('a group added after another sits directly below it, with a fresh id', () => {
    const id = addGroup('Beamline', 'workspace');
    expect(id).toMatch(/^g-[0-9a-f]{8}$/);
    expect(groupIds()).toEqual(['workspace', id, 'tools', 'system']);
    expect(labelOf(id)).toBe('Beamline');
  });

  test('a group added after no group goes last, and an empty name becomes the default', () => {
    const id = addGroup('   ');
    expect(groupIds()).toEqual(['workspace', 'tools', 'system', id]);
    expect(labelOf(id)).toBe(NEW_GROUP_LABEL);
  });

  test('every added group gets its own id', () => {
    const ids = new Set(Array.from({ length: 20 }, () => addGroup('Same')));
    expect(ids.size).toBe(20);
  });

  test('a built-in group is never deleted', () => {
    expect(removeGroup('workspace')).toBe(false);
    expect(removeGroup('tools')).toBe(false);
    expect(removeGroup('system')).toBe(false);
    expect(groupIds()).toEqual(['workspace', 'tools', 'system']);
  });

  test('deleting a group sends its panels back to Apps, Artifacts included', () => {
    const id = addGroup('Beamline');
    moveToGroup('ariel', id);
    moveToGroup('artifacts', id);
    moveToGroup('lattice', 'workspace');

    expect(removeGroup(id)).toBe(true);

    expect(groupIds()).toEqual(['workspace', 'tools', 'system']);
    expect(groupOf('ariel')).toBe('tools');
    expect(groupOf('artifacts')).toBe('tools');
    expect(groupOf('lattice')).toBe('workspace');
  });

  test('deleting an unknown group changes nothing', () => {
    expect(removeGroup('g-00000000')).toBe(false);
  });

  test('a group moves one place up or down, built-in ones too, never past an end', () => {
    const id = addGroup('Beamline');
    expect(reorderGroup(id, 'up')).toBe(true);
    expect(groupIds()).toEqual(['workspace', 'tools', id, 'system']);
    expect(reorderGroup('system', 'down')).toBe(false);
    expect(reorderGroup('tools', 'down')).toBe(true);
    expect(groupIds()).toEqual(['workspace', id, 'tools', 'system']);
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
      version: 3,
      groups: [
        { id: 'workspace', label: 'Sessions' },
        { id, label: 'Beamline' },
        { id: 'tools', label: 'Instruments' },
        { id: 'system', label: 'System' },
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
    expect(labelOf('tools')).toBe('Apps');
  });

  test('a value another tab stored is the base of the next change here', () => {
    renameGroup('tools', 'Mine');
    const other = defaultModel();
    other.groups[2].label = 'Theirs';
    localStorage.setItem(GROUPS_STORAGE_KEY, JSON.stringify(other));

    renameGroup('tools', 'Instruments');

    expect(groupLabels()).toEqual(['Sessions', 'Instruments', 'Theirs']);
  });

  const valid = defaultModel();
  test.each([
    ['not JSON', 'not json'],
    ['not an object', '[]'],
    ['null', 'null'],
    ['a later version', JSON.stringify({ ...valid, version: 4 })],
    ['a version that is not a number', JSON.stringify({ ...valid, version: '3' })],
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
      JSON.stringify({ ...valid, groups: [valid.groups[0], { id: 'tools', label: ' Apps' }] }),
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
    renameGroup('tools', 'Apps');

    expect(seen).toEqual([
      ['Sessions', 'Instruments', 'System'],
      ['Sessions', 'Instruments', 'System', 'Beamline'],
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

/** @param {HTMLElement} rail @param {string} id @returns {HTMLElement} */
const headingOf = (rail, id) => qs(groupEl(rail, id), ':scope > .sidebar-group-head');

/** @param {HTMLElement} rail @param {string} id @returns {HTMLElement} */
const nameOf = (rail, id) => qs(groupEl(rail, id), '.sidebar-group-name');

/** @returns {HTMLElement | null} the open group menu */
const menu = () => document.querySelector('.rail-context-menu');

/**
 * Right-click a group's heading at `target` (the heading itself by default).
 * @param {HTMLElement} rail @param {string} id @param {Element} [target]
 * @returns {MouseEvent} the event, to read whether the native menu was suppressed
 */
function rightClick(rail, id, target = headingOf(rail, id)) {
  const event = new MouseEvent('contextmenu', {
    bubbles: true,
    cancelable: true,
    clientX: 40,
    clientY: 50,
  });
  target.dispatchEvent(event);
  return event;
}

/** @returns {HTMLButtonElement[]} the rows of the open menu */
function menuItems() {
  const rows = menu()?.querySelectorAll('[role="menuitem"]') ?? [];
  return /** @type {HTMLButtonElement[]} */ ([...rows]);
}

/** @param {Element} row @returns {string} a menu row's words */
const rowLabel = (row) => qs(row, '.rail-context-label').textContent ?? '';

/** The rows of the open menu, as `label`, `label (disabled)` or `label (danger)`. */
function menuRows() {
  return menuItems().map((row) => {
    if (row.getAttribute('aria-disabled') === 'true') return `${rowLabel(row)} (disabled)`;
    return row.classList.contains('danger') ? `${rowLabel(row)} (danger)` : rowLabel(row);
  });
}

/** @param {string} label @returns {HTMLButtonElement} the open menu's row reading `label` */
function menuRow(label) {
  const row = menuItems().find((el) => rowLabel(el) === label);
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
  // Close whatever a test left open: the dismissal listeners are context-menu.js's.
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
  document.documentElement.removeAttribute('data-sidebar');
});

describe('group headings', () => {
  test('each heading holds only the group name, and names its group: no ⋯ button', () => {
    const rail = mountRail();
    for (const [id, label] of [
      ['workspace', 'Sessions'],
      ['tools', 'Apps'],
      ['system', 'System'],
    ]) {
      const group = groupEl(rail, id);
      const heading = headingOf(rail, id);
      expect(heading).toBe(group.firstElementChild);
      expect(heading.classList.contains('osprey-section-label')).toBe(true);
      const name = nameOf(rail, id);
      expect(name.textContent).toBe(label);
      expect([...heading.children]).toEqual([name]);
      expect(heading.querySelector('button')).toBeNull();
      expect(group.getAttribute('aria-labelledby')).toBe(name.id);
      expect(heading.hidden).toBe(id === 'workspace');
    }
  });

  test('the Sessions heading is hidden and opens no menu, no fold and no rename', () => {
    const rail = mountRail();
    const event = rightClick(rail, 'workspace');
    expect(event.defaultPrevented).toBe(false);
    expect(menu()).toBeNull();
    expect(startGroupRename(rail, 'workspace')).toBe(false);
    expect(renameField(rail, 'workspace')).toBeNull();
    // The Agent row still leads it.
    expect(groupEl(rail, 'workspace').querySelector('[data-panel-id="terminal"]')).not.toBeNull();
  });

  test('a renamed group shows its new name at once', () => {
    const rail = mountRail();
    renameGroup('tools', 'Instruments');
    expect(nameOf(rail, 'tools').textContent).toBe('Instruments');
  });
});

describe('the group menu', () => {
  test('a right-click on the heading opens a body-level menu: a built-in group has no Delete', () => {
    const rail = mountRail();
    const event = rightClick(rail, 'system');

    expect(event.defaultPrevented).toBe(true);
    const opened = /** @type {HTMLElement} */ (menu());
    expect(opened.parentElement).toBe(document.body);
    expect(opened.getAttribute('role')).toBe('menu');
    expect(opened.getAttribute('aria-label')).toBe('System group');
    expect(menuRows()).toEqual(['Rename', 'New group below', 'Move up', 'Move down (disabled)']);
  });

  test('the group below Sessions cannot move up past it', () => {
    const rail = mountRail();
    rightClick(rail, 'tools');
    expect(menuRows()).toEqual(['Rename', 'New group below', 'Move up (disabled)', 'Move down']);
  });

  test('the whole heading answers, not only its name: a right-click on the name opens it too', () => {
    const rail = mountRail();
    rightClick(rail, 'tools', nameOf(rail, 'tools'));
    expect(menu()?.getAttribute('aria-label')).toBe('Apps group');
  });

  test('the last group cannot move down, and a group the operator added can be deleted', () => {
    const rail = mountRail();
    const id = addGroup('Beamline');
    rightClick(rail, id);
    expect(menuRows()).toEqual([
      'Rename',
      'New group below',
      'Move up',
      'Move down (disabled)',
      'Delete group (danger)',
    ]);
  });

  test('a right-click on another heading moves the menu there', () => {
    const rail = mountRail();
    rightClick(rail, 'system');
    rightClick(rail, 'tools');
    expect(document.querySelectorAll('.rail-context-menu')).toHaveLength(1);
    expect(menu()?.getAttribute('aria-label')).toBe('Apps group');
  });

  for (const [what, init] of /** @type {const} */ ([
    ['Shift+F10', { key: 'F10', shiftKey: true }],
    ['the menu key', { key: 'ContextMenu' }],
  ])) {
    test(`${what} on the focused name opens the same menu`, () => {
      const rail = mountRail();
      const name = nameOf(rail, 'tools');
      const event = new KeyboardEvent('keydown', { ...init, bubbles: true, cancelable: true });
      name.dispatchEvent(event);
      expect(event.defaultPrevented).toBe(true);
      expect(menu()?.getAttribute('aria-label')).toBe('Apps group');
      expect(menuRows()).toEqual(['Rename', 'New group below', 'Move up (disabled)', 'Move down']);
    });
  }

  test('a rename field and the collapsed sidebar keep the browser menu', () => {
    const rail = mountRail();
    startGroupRename(rail, 'tools');
    const field = /** @type {HTMLInputElement} */ (renameField(rail, 'tools'));
    expect(rightClick(rail, 'tools', field).defaultPrevented).toBe(false);
    expect(menu()).toBeNull();
    press(field, 'Escape');

    document.documentElement.setAttribute('data-sidebar', 'collapsed');
    expect(rightClick(rail, 'tools').defaultPrevented).toBe(false);
    expect(menu()).toBeNull();
  });

  test('Move down moves the group, closes the menu and keeps focus on its name', () => {
    const rail = mountRail();
    rightClick(rail, 'tools');
    menuRow('Move down').click();

    expect(menu()).toBeNull();
    expect(groupIds()).toEqual(['workspace', 'system', 'tools']);
    const order = [...rail.children].map((el) => el.getAttribute('data-group'));
    expect(order).toEqual(['workspace', 'system', 'tools']);
    expect(document.activeElement).toBe(nameOf(rail, 'tools'));
  });

  test('Delete group removes it, and its rows go back to Apps', () => {
    const rail = mountRail();
    const id = addGroup('Beamline');
    moveToGroup('ariel', id);
    rightClick(rail, id);
    menuRow('Delete group').click();

    expect(groupIds()).toEqual(['workspace', 'tools', 'system']);
    expect(rail.querySelector(`[data-group="${id}"]`)).toBeNull();
    const ariel = rail.querySelector('[data-panel-id="ariel"]');
    expect(groupEl(rail, 'tools').contains(ariel)).toBe(true);
    expect(document.activeElement).toBe(nameOf(rail, 'tools'));
  });

  test('New group below adds an empty group under this one and opens its name', () => {
    const rail = mountRail();
    rightClick(rail, 'tools');
    menuRow('New group below').click();

    const ids = groupIds();
    expect(ids).toHaveLength(4);
    const id = ids[2];
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
    rightClick(rail, 'tools');
    menuRow('Rename').click();

    const field = renameField(rail, 'tools');
    const name = qs(groupEl(rail, 'tools'), '.sidebar-group-name');
    expect(field?.value).toBe('Apps');
    expect(field?.maxLength).toBe(40);
    expect(field?.getAttribute('aria-label')).toBe('Group name');
    expect(name.hidden).toBe(true);
    expect(document.activeElement).toBe(field);
  });

  test('Enter keeps the trimmed name and puts focus back on the name', () => {
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
    expect(document.activeElement).toBe(name);
  });

  test('Escape keeps the old name', () => {
    const rail = mountRail();
    startGroupRename(rail, 'tools');
    const field = /** @type {HTMLInputElement} */ (renameField(rail, 'tools'));
    field.value = 'Instruments';
    press(field, 'Escape');

    expect(labelOf('tools')).toBe('Apps');
    expect(renameField(rail, 'tools')).toBeNull();
    expect(document.activeElement).toBe(nameOf(rail, 'tools'));
  });

  test('leaving the field keeps the new name', () => {
    const rail = mountRail();
    startGroupRename(rail, 'tools');
    const field = /** @type {HTMLInputElement} */ (renameField(rail, 'tools'));
    field.value = 'Operations';
    field.dispatchEvent(new FocusEvent('blur'));

    expect(labelOf('tools')).toBe('Operations');
    expect(renameField(rail, 'tools')).toBeNull();
  });

  test('an empty field keeps the old name', () => {
    const rail = mountRail();
    startGroupRename(rail, 'tools');
    const field = /** @type {HTMLInputElement} */ (renameField(rail, 'tools'));
    field.value = '   ';
    press(field, 'Enter');

    expect(labelOf('tools')).toBe('Apps');
    expect(qs(groupEl(rail, 'tools'), '.sidebar-group-name').textContent).toBe('Apps');
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
