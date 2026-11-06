// @ts-check
/**
 * Compose autosave: compose-autosave.js keeps one browser-local snapshot per
 * compose context (C-AUTOSAVE) under one scoped storage record.
 *
 * Run: npx vitest run tests/interfaces/ariel/compose-autosave.test.mjs
 */
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const AUTOSAVE = '../../../src/osprey/interfaces/ariel/static/js/compose-autosave.js';

/** @type {typeof import('../../../src/osprey/interfaces/ariel/static/js/compose-autosave.js')} */
let autosave;

const DAY = 24 * 60 * 60 * 1000;

beforeEach(async () => {
  localStorage.clear();
  document.documentElement.removeAttribute('data-osprey-storage-scope');
  vi.resetModules();
  autosave = await import(AUTOSAVE);
});

/** @type {Array<{mockRestore: () => void}>} */
let spies = [];

afterEach(() => {
  // happy-dom's Storage is not reset by restoreAllMocks: undo each spy.
  for (const spy of spies) spy.mockRestore();
  spies = [];
  vi.useRealTimers();
});

/** @param {'getItem'|'setItem'|'removeItem'} method */
function block(method) {
  spies.push(
    vi.spyOn(localStorage, method).mockImplementation(() => {
      throw new Error('blocked');
    })
  );
}

/** @returns {any} */
function stored() {
  return JSON.parse(localStorage.getItem(autosave.AUTOSAVE_KEY) || 'null');
}

/** @param {Record<string, any>} [over] */
function snapshot(over = {}) {
  return {
    subject: 'Beam dump',
    details: 'Lost beam at 10:02.',
    author: 'kirk',
    fields: { system: 'RF', severity: '' },
    tags: ['rf', 'trip'],
    logbook: 'operations',
    shift: 'day',
    ...over,
  };
}

describe('contextKey', () => {
  test('a new entry', () => {
    expect(autosave.contextKey({ draft: null })).toBe('new');
  });

  test('a follow-up', () => {
    expect(autosave.contextKey({ draft: null, followsUp: 'e-42' })).toBe('follow:e-42');
  });

  test('a draft', () => {
    expect(autosave.contextKey({ draft: 'draft-0123456789ab' })).toBe('draft:draft-0123456789ab');
  });

  test('a draft wins over a stray followsUp', () => {
    expect(autosave.contextKey({ draft: 'draft-0123456789ab', followsUp: 'e-1' })).toBe(
      'draft:draft-0123456789ab'
    );
  });

  test('an empty followsUp is a new entry', () => {
    expect(autosave.contextKey({ draft: null, followsUp: '' })).toBe('new');
  });
});

describe('save, load, clear', () => {
  test('round trip', () => {
    expect(autosave.saveCompose('new', snapshot())).toBe(true);
    const loaded = autosave.loadCompose('new');
    expect(loaded).toMatchObject(snapshot());
    expect(typeof loaded?.savedAt).toBe('number');
  });

  test('a caller-supplied savedAt is kept', () => {
    const savedAt = Date.now() - 1000;
    autosave.saveCompose('new', snapshot({ savedAt }));
    expect(autosave.loadCompose('new')?.savedAt).toBe(savedAt);
  });

  test('blank declared fields are kept so a cleared field stays cleared', () => {
    autosave.saveCompose('new', snapshot({ fields: { system: '', severity: '' } }));
    expect(autosave.loadCompose('new')?.fields).toEqual({ system: '', severity: '' });
  });

  test('keys are independent and a snapshot never moves', () => {
    autosave.saveCompose('new', snapshot({ subject: 'A' }));
    autosave.saveCompose('follow:e-1', snapshot({ subject: 'B' }));
    autosave.saveCompose('draft:draft-0123456789ab', snapshot({ subject: 'C' }));
    expect(autosave.loadCompose('new')?.subject).toBe('A');
    expect(autosave.loadCompose('follow:e-1')?.subject).toBe('B');
    expect(autosave.loadCompose('draft:draft-0123456789ab')?.subject).toBe('C');
    expect(autosave.loadCompose('follow:e-2')).toBeNull();
  });

  test('all snapshots share one record', () => {
    autosave.saveCompose('new', snapshot());
    autosave.saveCompose('follow:e-1', snapshot());
    expect(Object.keys(stored()).sort()).toEqual(['follow:e-1', 'new']);
  });

  test('clear removes only its key', () => {
    autosave.saveCompose('new', snapshot());
    autosave.saveCompose('follow:e-1', snapshot());
    autosave.clearCompose('new');
    expect(autosave.loadCompose('new')).toBeNull();
    expect(autosave.loadCompose('follow:e-1')).not.toBeNull();
  });

  test('clearing the last key empties storage', () => {
    autosave.saveCompose('new', snapshot());
    autosave.clearCompose('new');
    expect(localStorage.getItem(autosave.AUTOSAVE_KEY)).toBeNull();
  });

  test('clearing a missing key is harmless', () => {
    expect(() => autosave.clearCompose('new')).not.toThrow();
    expect(autosave.loadCompose('new')).toBeNull();
  });

  test('loading with nothing stored returns null', () => {
    expect(autosave.loadCompose('new')).toBeNull();
  });

  test('the record honours the storage scope', () => {
    document.documentElement.setAttribute('data-osprey-storage-scope', 'acme');
    autosave.saveCompose('new', snapshot());
    expect(localStorage.getItem(`${autosave.AUTOSAVE_KEY}--acme`)).not.toBeNull();
    expect(localStorage.getItem(autosave.AUTOSAVE_KEY)).toBeNull();
    expect(autosave.loadCompose('new')?.subject).toBe('Beam dump');
  });
});

describe('30-day pruning', () => {
  test('saving prunes snapshots older than 30 days', () => {
    const now = Date.now();
    autosave.saveCompose('follow:e-old', snapshot({ savedAt: now - 31 * DAY }));
    autosave.saveCompose('follow:e-recent', snapshot({ savedAt: now - 29 * DAY }));
    autosave.saveCompose('new', snapshot());
    expect(Object.keys(stored()).sort()).toEqual(['follow:e-recent', 'new']);
  });

  test('a snapshot ages out between saves', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'));
    autosave.saveCompose('follow:e-1', snapshot());
    vi.setSystemTime(new Date('2026-02-15T00:00:00Z'));
    autosave.saveCompose('new', snapshot());
    expect(Object.keys(stored())).toEqual(['new']);
  });
});

describe('corrupt storage', () => {
  test('unparsable JSON loads as nothing and is replaced on save', () => {
    localStorage.setItem(autosave.AUTOSAVE_KEY, '{not json');
    expect(autosave.loadCompose('new')).toBeNull();
    expect(autosave.saveCompose('new', snapshot())).toBe(true);
    expect(autosave.loadCompose('new')?.subject).toBe('Beam dump');
  });

  test('a non-object record loads as nothing', () => {
    localStorage.setItem(autosave.AUTOSAVE_KEY, JSON.stringify(['new']));
    expect(autosave.loadCompose('new')).toBeNull();
  });

  test('a malformed snapshot loads as nothing and is dropped on save', () => {
    localStorage.setItem(
      autosave.AUTOSAVE_KEY,
      JSON.stringify({ 'follow:e-1': 'garbage', 'follow:e-2': { subject: 'x' } })
    );
    expect(autosave.loadCompose('follow:e-1')).toBeNull();
    expect(autosave.loadCompose('follow:e-2')).toBeNull();
    autosave.saveCompose('new', snapshot());
    expect(Object.keys(stored())).toEqual(['new']);
  });

  test('wrongly typed members are normalised on load', () => {
    localStorage.setItem(
      autosave.AUTOSAVE_KEY,
      JSON.stringify({
        new: {
          subject: 5,
          details: 'd',
          author: null,
          fields: ['x'],
          tags: ['ok', 3],
          logbook: 7,
          shift: 'night',
          savedAt: Date.now(),
        },
      })
    );
    expect(autosave.loadCompose('new')).toEqual({
      subject: '',
      details: 'd',
      author: '',
      fields: {},
      tags: ['ok'],
      logbook: null,
      shift: 'night',
      savedAt: expect.any(Number),
    });
  });
});

describe('storage that throws', () => {
  test('saveCompose returns false when setItem throws', () => {
    block('setItem');
    expect(autosave.saveCompose('new', snapshot())).toBe(false);
  });

  test('nothing throws when storage is blocked', () => {
    block('getItem');
    block('setItem');
    block('removeItem');
    expect(autosave.saveCompose('new', snapshot())).toBe(false);
    expect(autosave.loadCompose('new')).toBeNull();
    expect(() => autosave.clearCompose('new')).not.toThrow();
  });

  test('a bad key or snapshot returns false', () => {
    // @ts-expect-error deliberate misuse
    expect(autosave.saveCompose('', snapshot())).toBe(false);
    // @ts-expect-error deliberate misuse
    expect(autosave.saveCompose('new', null)).toBe(false);
    expect(localStorage.getItem(autosave.AUTOSAVE_KEY)).toBeNull();
  });
});

describe('credential and attachment fields', () => {
  test('are never stored', () => {
    autosave.saveCompose(
      'new',
      snapshot({
        auth_username: 'kirk',
        auth_password: 'hunter2',
        password: 'hunter2',
        attachments: [{ name: 'a.png' }],
        attachment_paths: ['/tmp/a.png'],
        files: ['a.png'],
        fields: {
          system: 'RF',
          auth_password: 'hunter2',
          auth_username: 'kirk',
          attachments: 'a.png',
        },
      })
    );
    const raw = localStorage.getItem(autosave.AUTOSAVE_KEY) || '';
    expect(raw).not.toContain('hunter2');
    expect(raw).not.toContain('a.png');
    expect(raw).not.toContain('auth_');
    const loaded = autosave.loadCompose('new');
    expect(Object.keys(loaded || {}).sort()).toEqual([
      'author',
      'details',
      'fields',
      'logbook',
      'savedAt',
      'shift',
      'subject',
      'tags',
    ]);
    expect(loaded?.fields).toEqual({ system: 'RF' });
  });

  test('are stripped from records that already hold them', () => {
    localStorage.setItem(
      autosave.AUTOSAVE_KEY,
      JSON.stringify({
        new: { ...snapshot(), auth_password: 'hunter2', savedAt: Date.now() },
      })
    );
    expect(autosave.loadCompose('new')).not.toHaveProperty('auth_password');
  });
});
