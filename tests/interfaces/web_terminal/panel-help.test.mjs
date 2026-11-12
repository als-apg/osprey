/**
 * Unit tests for the panel help map (panel-help.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-help.test.mjs
 *
 * Besides the URL join, one test walks every page in HELP_PAGES and checks that
 * its `.rst` source exists under docs/source/, so a renamed or deleted page
 * fails here instead of becoming a dead Help link.
 */

import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { test, expect, describe, afterEach, vi } from 'vitest';

import {
  HELP_PAGES,
  docsRoot,
  helpUrl,
  openHelp,
} from '../../../src/osprey/interfaces/web_terminal/static/js/panel-help.js';
import {
  PANELS,
  customPanelEntry,
} from '../../../src/osprey/interfaces/web_terminal/static/js/panel-catalog.js';

// node:path, not the global URL: happy-dom replaces URL and rejects file: bases.
const DOCS_SOURCE = resolve(dirname(fileURLToPath(import.meta.url)), '../../../docs/source');

afterEach(() => {
  delete document.documentElement.dataset.docsUrl;
  for (let i = PANELS.length - 1; i >= 0; i--) {
    if (PANELS[i].id === 'dial' || PANELS[i].id === 'manual') PANELS.splice(i, 1);
  }
  vi.restoreAllMocks();
});

describe('HELP_PAGES', () => {
  test('is frozen', () => {
    expect(Object.isFrozen(HELP_PAGES)).toBe(true);
  });

  test.each(Object.entries(HELP_PAGES))('%s → %s has a source page', (_id, page) => {
    const source = page.replace(/#.*$/, '').replace(/\.html$/, '.rst');
    expect(page.endsWith('.html') || page.includes('.html#')).toBe(true);
    expect(existsSync(resolve(DOCS_SOURCE, source))).toBe(true);
  });
});

describe('docsRoot', () => {
  test('is the trimmed data-docs-url', () => {
    document.documentElement.dataset.docsUrl = '  https://docs.example/osprey/ ';
    expect(docsRoot()).toBe('https://docs.example/osprey/');
  });

  test('is empty when the attribute is absent or blank', () => {
    expect(docsRoot()).toBe('');
    document.documentElement.dataset.docsUrl = '   ';
    expect(docsRoot()).toBe('');
  });
});

describe('helpUrl', () => {
  test('joins with one slash whether or not the root ends with one', () => {
    const expected = 'https://docs.example/osprey/how-to/use-channel-finder.html';
    expect(helpUrl('channel-finder', 'https://docs.example/osprey')).toBe(expected);
    expect(helpUrl('channel-finder', 'https://docs.example/osprey/')).toBe(expected);
  });

  test('keeps the anchor of a page that has one', () => {
    expect(helpUrl('events', 'https://d.example/')).toBe(
      'https://d.example/how-to/agent-interfaces/event-dispatch.html#the-events-panel'
    );
  });

  test('is null for a blank root', () => {
    expect(helpUrl('ariel', '')).toBeNull();
    expect(helpUrl('ariel', '  ')).toBeNull();
    expect(helpUrl('ariel')).toBeNull();
  });

  test('is null for a panel with no page, including inherited names', () => {
    expect(helpUrl('no-such-panel', 'https://d.example')).toBeNull();
    expect(helpUrl('toString', 'https://d.example')).toBeNull();
  });

  test('an application panel declares its own page under the docs root', () => {
    PANELS.push(
      customPanelEntry({ id: 'dial', label: 'DIAL', help: 'how-to/apps/build-your-own-app.html' })
    );
    expect(helpUrl('dial', 'https://d.example/')).toBe(
      'https://d.example/how-to/apps/build-your-own-app.html'
    );
    expect(helpUrl('dial', '')).toBeNull();
  });

  test('an application panel may declare an absolute https page, with or without a docs root', () => {
    const page = 'https://example.org/dial/manual.html#start';
    PANELS.push(customPanelEntry({ id: 'manual', label: 'MANUAL', help: page }));
    expect(helpUrl('manual', 'https://d.example/')).toBe(page);
    expect(helpUrl('manual', '')).toBe(page);
  });

  test('a built-in panel keeps its mapped page', () => {
    PANELS.push(customPanelEntry({ id: 'dial', label: 'DIAL', help: 'other.html' }));
    expect(helpUrl('ariel', 'https://d.example')).toBe(
      'https://d.example/how-to/ariel/web-interface.html'
    );
  });

  test('reads the root from the page by default', () => {
    document.documentElement.dataset.docsUrl = 'https://d.example/';
    expect(helpUrl('ariel')).toBe('https://d.example/how-to/ariel/web-interface.html');
  });
});

describe('openHelp', () => {
  test('opens the page in a new tab without an opener', () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null);
    document.documentElement.dataset.docsUrl = 'https://d.example';
    expect(openHelp('jupyter')).toBe(true);
    expect(open).toHaveBeenCalledWith(
      'https://d.example/how-to/web-terminal/notebooks.html',
      '_blank',
      'noopener'
    );
  });

  test('is false without a docs root or a page', () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null);
    expect(openHelp('jupyter')).toBe(false);
    document.documentElement.dataset.docsUrl = 'https://d.example';
    expect(openHelp('no-such-panel')).toBe(false);
    expect(open).not.toHaveBeenCalled();
  });
});
