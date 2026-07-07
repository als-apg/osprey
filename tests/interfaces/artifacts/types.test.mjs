/**
 * Unit tests for the Artifact Gallery type-registry/formatting/color-pass
 * layer (types.js).
 *
 * Pure-logic/DOM guard, happy-dom environment (configured globally), `fetch`
 * mocked via vi.stubGlobal where needed:
 *   npx vitest run tests/interfaces/artifacts/types.test.mjs
 *
 * Covers typeColor/typeBadge/typeIcon for known + unknown types,
 * escapeHtml's null-guard contract, formatSize/formatTime/formatFullTime/
 * formatDate boundary values, openUrl's per-type routing, thumbnailHtml,
 * isNewThisSession (with the session-start now an explicit parameter),
 * initTypeRegistry/getTypeRegistry, and the color-pass DOM helper.
 *
 * NOTE: imported by RELATIVE path — this module lives under artifacts, not
 * design-system, so no alias applies.
 */

import { test, expect, vi, describe, afterEach } from 'vitest';

import {
  getTypeRegistry,
  initTypeRegistry,
  typeBadge,
  typeIcon,
  typeColor,
  thumbnailHtml,
  escapeHtml,
  formatSize,
  formatTime,
  formatFullTime,
  formatDate,
  openUrl,
  isNewThisSession,
  sendToTerminal,
  requestColorPass,
} from '../../../src/osprey/interfaces/artifacts/static/js/types.js';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('typeColor', () => {
  test('an unknown type falls back to the theme-invariant default', () => {
    expect(typeColor('totally_unknown_type')).toBe('#64748b');
  });

  test('a registered artifact_type returns its registry color', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ artifact_types: { plot_html: { label: 'Plot', color: '#ff0000' } } }),
    }));
    await initTypeRegistry();
    expect(typeColor('plot_html')).toBe('#ff0000');
  });

  test('categories take precedence over artifact_types when both are present', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: () => Promise.resolve({
        categories: { visualization: { label: 'Viz', color: '#00ff00' } },
        artifact_types: { visualization: { label: 'Viz (type)', color: '#0000ff' } },
      }),
    }));
    await initTypeRegistry();
    expect(typeColor('visualization')).toBe('#00ff00');
    expect(typeBadge('visualization')).toBe('Viz');
  });
});

describe('typeBadge', () => {
  test('an unknown type falls back to a humanized version of the raw type string', () => {
    expect(typeBadge('some_custom_type')).toBe('some custom type');
  });

  test('a registered type returns its registry label', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ artifact_types: { json: { label: 'JSON Data' } } }),
    }));
    await initTypeRegistry();
    expect(typeBadge('json')).toBe('JSON Data');
  });
});

describe('typeIcon', () => {
  test('a known type returns its own SVG icon', () => {
    expect(typeIcon('markdown')).toContain('<svg');
    expect(typeIcon('markdown')).not.toBe(typeIcon('plot_html'));
  });

  test('an unknown type falls back to the generic text icon', () => {
    expect(typeIcon('some_unregistered_type')).toBe(typeIcon('text'));
  });
});

describe('initTypeRegistry / getTypeRegistry', () => {
  test('populates the registry from /api/type-registry on success', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ artifact_types: { html: { label: 'HTML' } } }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await initTypeRegistry();
    expect(fetchMock).toHaveBeenCalledWith('/api/type-registry');
    expect(getTypeRegistry()).toEqual({ artifact_types: { html: { label: 'HTML' } } });
  });

  test('is silent (does not throw) on a network failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network down')));
    await expect(initTypeRegistry()).resolves.toBeUndefined();
  });
});

describe('escapeHtml', () => {
  test('null and undefined both yield the empty string (null-guard contract)', () => {
    expect(escapeHtml(null)).toBe('');
    expect(escapeHtml(undefined)).toBe('');
  });

  test('escapes angle brackets and ampersands', () => {
    expect(escapeHtml('<b>&')).toBe('&lt;b&gt;&amp;');
  });

  test('escapes a <script> tag', () => {
    expect(escapeHtml('<script>alert(1)</script>')).toBe('&lt;script&gt;alert(1)&lt;/script&gt;');
  });

  test('an empty string stays empty', () => {
    expect(escapeHtml('')).toBe('');
  });

  test('KNOWN DIVERGENCE from design-system/js/dom.js\'s escapeHtml: falsy-but-defined values (0, false) also collapse to "" here', () => {
    // dom.js's escapeHtml uses `String(value ?? "")` — only null/undefined
    // become "", so escapeHtml(0) there is "0" and escapeHtml(false) is
    // "false". This local implementation uses `str || ""` (moved verbatim
    // from the original gallery.js, unchanged behavior), which treats ANY
    // falsy value the same as nullish. Pinning the current behavior here so
    // a future consolidation onto the shared dom.js helper is a deliberate,
    // reviewed decision rather than a silent behavior change.
    expect(escapeHtml(0)).toBe('');
    expect(escapeHtml(false)).toBe('');
  });
});

describe('formatSize', () => {
  test('0 or falsy bytes renders as "0 B"', () => {
    expect(formatSize(0)).toBe('0 B');
    expect(formatSize(null)).toBe('0 B');
    expect(formatSize(undefined)).toBe('0 B');
  });

  test('sub-1024 bytes stay in B, whole numbers', () => {
    expect(formatSize(1)).toBe('1 B');
    expect(formatSize(1023)).toBe('1023 B');
  });

  test('boundary: exactly 1024 bytes rolls over to 1.0 KB', () => {
    expect(formatSize(1024)).toBe('1.0 KB');
  });

  test('boundary: exactly 1024*1024 bytes rolls over to 1.0 MB', () => {
    expect(formatSize(1024 * 1024)).toBe('1.0 MB');
  });

  test('caps at GB (does not roll over past the largest unit)', () => {
    expect(formatSize(1024 * 1024 * 1024 * 5)).toBe('5.0 GB');
  });
});

describe('formatTime / formatFullTime / formatDate', () => {
  test('formatTime and formatFullTime return "" for falsy input', () => {
    expect(formatTime(null)).toBe('');
    expect(formatTime(undefined)).toBe('');
    expect(formatFullTime(null)).toBe('');
    expect(formatFullTime(undefined)).toBe('');
  });

  test('formatDate returns "Unknown" for falsy input', () => {
    expect(formatDate(null)).toBe('Unknown');
    expect(formatDate(undefined)).toBe('Unknown');
  });

  test('formatDate returns "Today" for the current date', () => {
    expect(formatDate(new Date().toISOString())).toBe('Today');
  });

  test('formatDate returns "Yesterday" for the prior calendar day', () => {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    expect(formatDate(yesterday.toISOString())).toBe('Yesterday');
  });

  test('formatTime and formatFullTime produce non-empty output for a valid ISO string', () => {
    const iso = '2026-07-03T15:45:00Z';
    expect(formatTime(iso).length).toBeGreaterThan(0);
    expect(formatFullTime(iso)).toContain('2026');
  });
});

describe('openUrl', () => {
  test('markdown routes to the rendered-markdown API endpoint', () => {
    expect(openUrl({ id: 'a1', artifact_type: 'markdown', filename: 'x.md' })).toBe('/api/markdown/a1/rendered');
  });

  test('notebook routes to the rendered-notebook API endpoint', () => {
    expect(openUrl({ id: 'a2', artifact_type: 'notebook', filename: 'x.ipynb' })).toBe('/api/notebooks/a2/rendered');
  });

  test('any other type falls back to the raw file URL', () => {
    expect(openUrl({ id: 'a3', artifact_type: 'plot_png', filename: 'plot.png' })).toBe('/files/a3/plot.png');
  });
});

describe('thumbnailHtml', () => {
  test('an image type renders an <img> tag pointing at the file URL', () => {
    const html = thumbnailHtml({ id: 'i1', artifact_type: 'image', filename: 'photo.png' });
    expect(html).toContain('<img');
    expect(html).toContain('/files/i1/photo.png');
  });

  test('an html-family type renders an <iframe>', () => {
    const html = thumbnailHtml({ id: 'i2', artifact_type: 'plot_html', filename: 'p.html' });
    expect(html).toContain('<iframe');
  });

  test('a type with a non-empty summary renders the summary fields, escaped', () => {
    const html = thumbnailHtml({ id: 'i3', artifact_type: 'json', filename: 'd.json', summary: { rows: 10 } });
    expect(html).toContain('thumb-summary');
    expect(html).toContain('rows: 10');
  });

  test('a type with no summary falls back to the generic type icon + badge', () => {
    const html = thumbnailHtml({ id: 'i4', artifact_type: 'unknown_type', filename: 'f.bin' });
    expect(html).toContain('thumb-placeholder');
  });
});

describe('isNewThisSession', () => {
  test('an artifact timestamped at/after session start is new', () => {
    expect(isNewThisSession({ timestamp: '2026-07-03T12:00:00Z' }, '2026-07-03T12:00:00Z')).toBe(true);
    expect(isNewThisSession({ timestamp: '2026-07-03T13:00:00Z' }, '2026-07-03T12:00:00Z')).toBe(true);
  });

  test('an artifact timestamped before session start is not new', () => {
    expect(isNewThisSession({ timestamp: '2026-07-03T11:00:00Z' }, '2026-07-03T12:00:00Z')).toBe(false);
  });

  test('an artifact with no timestamp is not new', () => {
    expect(isNewThisSession({}, '2026-07-03T12:00:00Z')).toBe(false);
  });
});

describe('sendToTerminal', () => {
  test('is a no-op outside an embedded/iframed context (window.parent === window)', () => {
    expect(() => sendToTerminal('hello')).not.toThrow();
  });
});

describe('requestColorPass', () => {
  test('recolors a badge-* element on the next animation frame, without throwing', async () => {
    document.body.innerHTML = '<span class="badge-plot_html"></span>';
    requestColorPass();
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const el = document.querySelector('.badge-plot_html');
    expect(el.style.color).not.toBe('');
  });
});
