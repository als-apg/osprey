// @ts-nocheck
// TODO(frontend-hardening): type-clean this test; tracked in eslint.config.js local/no-ts-nocheck allowlist, which may only shrink.
/**
 * Unit tests for settings-editor.js -- the interactive settings.json editor
 * behind config-renderers.js's re-export. Covers three pinned behaviors:
 *
 *   - the editor emits a dirty-change callback on mutation (model field
 *     edits, permission entry deletion)
 *   - drag-and-drop entry IDs (`_nextDragId`) are unique, including across
 *     multiple render calls in the same session (module-level counter)
 *   - getData() round-trips permission entries back into a settings.json
 *     shape, and correctly excludes entries mid-removal-animation
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/settings-editor.test.mjs
 *
 * settings-editor.js keeps `_nextDragId` as a module-level singleton (like
 * theme-manager.js's role/preference state), so each test resets the module
 * registry and re-imports fresh -- see theme-settheme.test.mjs for the same
 * pattern.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

const SETTINGS_JSON = JSON.stringify({
  model: 'anthropic/claude-sonnet',
  permissions: {
    allow: ['Bash', 'Read(foo)'],
    ask: ['mcp__scan__start'],
    deny: ['Task(danger-agent)'],
  },
}, null, 2);

describe('renderSettingsJsonEditor', () => {
  /** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/settings-editor.js')} */
  let SettingsEditor;

  beforeEach(async () => {
    vi.resetModules();
    SettingsEditor = await import(
      '../../../src/osprey/interfaces/web_terminal/static/js/settings-editor.js'
    );
    document.body.innerHTML = '';
  });

  test('returns null on invalid JSON (matches renderSettingsJson\'s parse-guard)', () => {
    expect(SettingsEditor.renderSettingsJsonEditor('not json', () => {})).toBeNull();
  });

  describe('dirty-change emission', () => {
    test('is not dirty immediately after render', () => {
      const container = SettingsEditor.renderSettingsJsonEditor(SETTINGS_JSON, () => {});
      expect(container._settingsEditor.isDirty()).toBe(false);
    });

    test('editing the model field marks dirty and invokes the callback', () => {
      const onDirtyChange = vi.fn();
      const container = SettingsEditor.renderSettingsJsonEditor(SETTINGS_JSON, onDirtyChange);

      container._modelInput.value = 'anthropic/claude-opus';
      container._modelInput.dispatchEvent(new Event('input'));

      expect(onDirtyChange).toHaveBeenCalledWith(true);
      expect(container._settingsEditor.isDirty()).toBe(true);
    });

    test('deleting a permission entry marks dirty once the removal animation completes', async () => {
      const onDirtyChange = vi.fn();
      const container = SettingsEditor.renderSettingsJsonEditor(SETTINGS_JSON, onDirtyChange);

      const deleteBtn = container.querySelector('.config-perm-entry-delete');
      expect(deleteBtn).not.toBeNull();
      deleteBtn.click();

      // The delete handler defers markDirty() by 200ms (removal animation).
      expect(onDirtyChange).not.toHaveBeenCalled();
      await new Promise((resolve) => setTimeout(resolve, 220));

      expect(onDirtyChange).toHaveBeenCalledWith(true);
      expect(container._settingsEditor.isDirty()).toBe(true);
    });
  });

  describe('drag-id uniqueness', () => {
    test('every rendered permission entry gets a unique data-drag-id', () => {
      const container = SettingsEditor.renderSettingsJsonEditor(SETTINGS_JSON, () => {});
      const ids = Array.from(
        container.querySelectorAll('.config-perm-entry-interactive')
      ).map((e) => e.dataset.dragId);

      expect(ids.length).toBe(4); // Bash, Read(foo), mcp__scan__start, Task(danger-agent)
      expect(new Set(ids).size).toBe(ids.length);
    });

    test('ids stay unique across multiple render calls in the same session', () => {
      const first = SettingsEditor.renderSettingsJsonEditor(SETTINGS_JSON, () => {});
      const second = SettingsEditor.renderSettingsJsonEditor(SETTINGS_JSON, () => {});

      const idsOf = (container) =>
        Array.from(container.querySelectorAll('.config-perm-entry-interactive')).map(
          (e) => e.dataset.dragId
        );
      const allIds = [...idsOf(first), ...idsOf(second)];

      expect(new Set(allIds).size).toBe(allIds.length);
    });
  });

  describe('drag-and-drop wiring', () => {
    // Regression guard for the pre-existing `_wireDragAndDrop(columns, colMap,
    // markDirty, container)` 4-args-vs-3-params call-site bug fixed alongside
    // this split (the extraneous 4th arg was always silently ignored by JS;
    // dropping it must not change drag-and-drop behavior).
    function makeDataTransfer() {
      const store = new Map();
      return {
        effectAllowed: '',
        dropEffect: '',
        setData: (type, val) => store.set(type, val),
        getData: (type) => store.get(type) ?? '',
      };
    }

    test('dragging an entry into another column moves it and marks dirty', () => {
      const onDirtyChange = vi.fn();
      const container = SettingsEditor.renderSettingsJsonEditor(SETTINGS_JSON, onDirtyChange);

      const allowCol = container.querySelector('.config-perm-allow');
      const askCol = container.querySelector('.config-perm-ask');
      const entry = allowCol.querySelector('.config-perm-entry-interactive'); // 'Bash'

      const dataTransfer = makeDataTransfer();

      const dragStartEvent = new Event('dragstart', { bubbles: true });
      dragStartEvent.dataTransfer = dataTransfer;
      entry.dispatchEvent(dragStartEvent);

      const dropEvent = new Event('drop', { bubbles: true });
      dropEvent.dataTransfer = dataTransfer;
      askCol.dispatchEvent(dropEvent);

      expect(askCol.contains(entry)).toBe(true);
      expect(allowCol.contains(entry)).toBe(false);
      expect(onDirtyChange).toHaveBeenCalledWith(true);
    });
  });

  describe('permission-entry serialization round-trip', () => {
    test('getData() round-trips model and permission entries unmodified', () => {
      const container = SettingsEditor.renderSettingsJsonEditor(SETTINGS_JSON, () => {});
      const result = JSON.parse(container._settingsEditor.getData());

      expect(result.model).toBe('anthropic/claude-sonnet');
      expect(result.permissions.allow.slice().sort()).toEqual(['Bash', 'Read(foo)'].sort());
      expect(result.permissions.ask).toEqual(['mcp__scan__start']);
      expect(result.permissions.deny).toEqual(['Task(danger-agent)']);
    });

    test('getData() excludes an entry mid-removal, even before its timeout fires', () => {
      const container = SettingsEditor.renderSettingsJsonEditor(SETTINGS_JSON, () => {});

      // First rendered draggable entry is 'Bash' (ungrouped entries render
      // before the grouped 'file access' entry, per _groupPermissions).
      const deleteBtn = container.querySelector('.config-perm-entry-delete');
      deleteBtn.click(); // synchronously adds .config-perm-removing

      const result = JSON.parse(container._settingsEditor.getData());
      expect(result.permissions.allow).not.toContain('Bash');
      expect(result.permissions.allow).toContain('Read(foo)');
    });

    test('getData() reflects a completed deletion', async () => {
      const container = SettingsEditor.renderSettingsJsonEditor(SETTINGS_JSON, () => {});

      const deleteBtn = container.querySelector('.config-perm-entry-delete');
      deleteBtn.click();
      await new Promise((resolve) => setTimeout(resolve, 220));

      const result = JSON.parse(container._settingsEditor.getData());
      expect(result.permissions.allow).toEqual(['Read(foo)']);
    });
  });
});
