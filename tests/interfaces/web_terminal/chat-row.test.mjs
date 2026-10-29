/**
 * Unit tests for chat-row.js -- the shape every action row shares.
 *
 * Glyph, verb phrase, detail, status, in that order; the glyph is decorative
 * and the state drives the status mark. A row settles without changing shape.
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-row.test.mjs
 */

import { describe, test, expect } from 'vitest';

import { qs } from '../_support/dom.mjs';

import { chatIcon } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-icons.js';
import {
  createRow,
  iconStack,
  statusMark,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-row.js';

describe('createRow', () => {
  test('lays the row out as glyph, phrase, detail, status, with the body hidden', () => {
    const row = createRow({
      className: 'op-tool',
      kind: 'read',
      glyph: chatIcon('read'),
      label: 'Read',
      detail: 'orbit.py',
    });

    expect(row.el.className).toBe('op-row op-tool');
    expect(row.el.dataset.kind).toBe('read');
    expect([...row.head.children].map((el) => el.className)).toEqual([
      'op-row-gutter',
      'op-row-label',
      'op-row-detail',
      'op-row-status',
    ]);
    expect(row.head.tagName).toBe('BUTTON');
    expect(row.head.getAttribute('aria-expanded')).toBe('false');
    expect(qs(row.el, '.op-row-glyph .op-icon').getAttribute('aria-hidden')).toBe('true');
    expect(row.label.textContent).toBe('Read');
    expect(row.detail.textContent).toBe('orbit.py');
    expect(row.body.hidden).toBe(true);
  });

  test('starts running with a spinner and settles to a mark of the same slot', () => {
    const row = createRow({
      className: 'op-tool',
      kind: 'bash',
      glyph: chatIcon('bash'),
      label: 'Bash',
    });
    expect(row.el.dataset.state).toBe('running');
    expect(row.status.querySelector('.op-spin')).not.toBeNull();

    row.setState('done');
    expect(row.el.dataset.state).toBe('done');
    expect(row.status.querySelector('.op-spin')).toBeNull();
    expect(row.status.querySelector('.op-icon')).not.toBeNull();
    expect(row.status.children).toHaveLength(1);

    row.setState('error');
    expect(row.el.dataset.state).toBe('error');
    expect(row.status.children).toHaveLength(1);
  });

  test('a row that keeps its own status is left alone by the state', () => {
    const row = createRow({
      className: 'op-thinking',
      kind: 'thinking',
      glyph: chatIcon('brain'),
      label: 'Thinking',
      ownStatus: true,
    });
    row.status.append(chatIcon('chevron'));
    row.setState('done');
    expect(row.status.children).toHaveLength(1);
    expect(row.status.querySelector('.op-spin')).toBeNull();
  });

  test('opens and closes, and says so', () => {
    const row = createRow({
      className: 'op-tool',
      kind: 'read',
      glyph: chatIcon('read'),
      label: 'Read',
    });
    row.setOpen(true);
    expect(row.isOpen()).toBe(true);
    expect(row.body.hidden).toBe(false);
    expect(row.el.classList.contains('open')).toBe(true);
    expect(row.head.getAttribute('aria-expanded')).toBe('true');
    row.setOpen(false);
    expect(row.isOpen()).toBe(false);
    expect(row.head.getAttribute('aria-expanded')).toBe('false');
  });

  test('a stack of glyphs marks the row as a run', () => {
    const row = createRow({
      className: 'op-run',
      kind: 'run',
      glyph: iconStack('read', 7),
      label: 'Read 7 files',
    });
    expect(row.glyph.classList.contains('op-row-stack')).toBe(true);
    expect(row.glyph.querySelectorAll('.op-icon')).toHaveLength(4);
  });
});

describe('statusMark', () => {
  test.each([
    ['running', '.op-spin'],
    ['background', '.op-spin'],
    ['done', 'svg'],
    ['error', 'svg'],
    ['stopped', 'svg'],
  ])('%s is drawn as %s', (state, selector) => {
    const mark = statusMark(/** @type {any} */ (state));
    expect(mark.matches(selector)).toBe(true);
  });
});
