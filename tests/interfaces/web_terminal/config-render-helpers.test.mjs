/**
 * Unit tests for config-render-helpers.js's `_renderHookEvents` -- the
 * per-event hooks tree shared by the read-only settings.json renderer
 * (config-renderers.js) and the editor's read-only Hooks section
 * (settings-editor.js). Pins the DOM contract both consumers rely on:
 *
 *   - one `.config-hook-event` per event, header with chevron + event name +
 *     count badge (total hooks across matcher groups)
 *   - clicking the header toggles the `expanded` class
 *   - matcher label falls back to `*`; script names are derived from the
 *     command path (basename, quotes stripped, `.py` dropped)
 *   - a timeout badge appears only when the hook declares one
 *   - all text lands via textContent (no HTML sink: markup-ish event names
 *     must not become elements)
 *
 * Pure DOM guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/config-render-helpers.test.mjs
 */

import { test, expect, describe } from 'vitest';

import { qs } from '../_support/dom.mjs';
import { _renderHookEvents } from '../../../src/osprey/interfaces/web_terminal/static/js/config-render-helpers.js';

/** Render into a container so class queries work on a rooted tree.
 * @param {Record<string, Array<{matcher?: string, hooks?: Array<{command?: string, timeout?: number}>}>>} hooks
 */
function renderHooks(hooks) {
  const container = document.createElement('div');
  container.appendChild(_renderHookEvents(hooks));
  return container;
}

describe('_renderHookEvents', () => {
  test('renders one event section per event with the total hook count in the badge', () => {
    const container = renderHooks({
      PreToolUse: [
        { matcher: 'Bash', hooks: [{ command: 'a.py' }, { command: 'b.py' }] },
        { hooks: [{ command: 'c.py' }] },
      ],
      Stop: [{ hooks: [{ command: 'd.py' }] }],
    });

    const events = container.querySelectorAll('.config-hook-event');
    expect(events).toHaveLength(2);
    expect(qs(events[0], '.config-hook-count').textContent).toBe('3');
    expect(qs(events[1], '.config-hook-count').textContent).toBe('1');
  });

  test('clicking the event header toggles the expanded class', () => {
    const container = renderHooks({ PreToolUse: [{ hooks: [{ command: 'a.py' }] }] });
    const event = qs(container, '.config-hook-event');

    expect(event.classList.contains('expanded')).toBe(false);
    qs(event, '.config-hook-event-header').dispatchEvent(new Event('click'));
    expect(event.classList.contains('expanded')).toBe(true);
    qs(event, '.config-hook-event-header').dispatchEvent(new Event('click'));
    expect(event.classList.contains('expanded')).toBe(false);
  });

  test('matcher label falls back to `*`; script name is the de-quoted basename without .py', () => {
    const container = renderHooks({
      PreToolUse: [{
        hooks: [{ command: 'uv run "/opt/osprey/.claude/hooks/write_guard.py"' }],
      }],
    });

    expect(qs(container, '.config-hook-matcher-label').textContent).toBe('*');
    expect(qs(container, '.config-hook-script').textContent).toBe('write_guard');
  });

  test('timeout badge appears only when the hook declares a timeout', () => {
    const container = renderHooks({
      PreToolUse: [{ hooks: [{ command: 'a.py', timeout: 30 }, { command: 'b.py' }] }],
    });

    const entries = container.querySelectorAll('.config-hook-entry');
    expect(qs(entries[0], '.config-hook-timeout').textContent).toBe('30s');
    expect(entries[1].querySelector('.config-hook-timeout')).toBeNull();
  });

  test('event names land via textContent, never as markup', () => {
    const container = renderHooks({ '<img src=x onerror=hack()>': [{ hooks: [] }] });

    expect(container.querySelector('img')).toBeNull();
    const nameSpan = qs(container, '.config-hook-event-header').children[1];
    expect(nameSpan.textContent).toBe('<img src=x onerror=hack()>');
  });
});
