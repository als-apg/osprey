/**
 * Unit tests for the design-system micro-frontend query-param contract.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/frame-params.test.js
 *
 * Covers CONTRACT_VERSION, applyEmbedded() (the embedded class, and on a
 * pop-out page the `data-popout` stamp and the pop-out client start),
 * isEmbedded(), isPopout() and terminalOffered().
 *
 * NOTE: frame-params.js is imported by RELATIVE path, not the absolute
 * `/design-system/js/frame-params.js` runtime specifier — matching the
 * sibling suites in this directory (see
 * tests/interfaces/design_system/js/dom.test.js).
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

const { initPopoutClient } = vi.hoisted(() => ({ initPopoutClient: vi.fn(() => true) }));
vi.mock('/design-system/js/popout-client.js', () => ({ initPopoutClient }));

import {
  applyEmbedded,
  terminalOffered,
  isEmbedded,
  isPopout,
  CONTRACT_VERSION,
} from '../../../../src/osprey/interfaces/design_system/static/js/frame-params.js';

describe('applyEmbedded', () => {
  afterEach(() => {
    document.body.className = '';
  });

  test('?embedded=true adds the embedded class to document.body', () => {
    window.history.replaceState({}, '', '?embedded=true');

    applyEmbedded();

    expect(document.body.classList.contains('embedded')).toBe(true);
  });

  test('no embedded param leaves the embedded class absent', () => {
    window.history.replaceState({}, '', '?');

    applyEmbedded();

    expect(document.body.classList.contains('embedded')).toBe(false);
  });

  test('?embedded=false leaves the embedded class absent', () => {
    window.history.replaceState({}, '', '?embedded=false');

    applyEmbedded();

    expect(document.body.classList.contains('embedded')).toBe(false);
  });

  test('?embedded=1 leaves the embedded class absent', () => {
    window.history.replaceState({}, '', '?embedded=1');

    applyEmbedded();

    expect(document.body.classList.contains('embedded')).toBe(false);
  });

  test('?popout=true stamps data-popout on <html> and starts the client', () => {
    window.history.replaceState({}, '', '?popout=true');
    initPopoutClient.mockClear();

    applyEmbedded();

    expect(document.documentElement.getAttribute('data-popout')).toBe('true');
    expect(document.body.classList.contains('embedded')).toBe(false);
    expect(initPopoutClient).toHaveBeenCalledTimes(1);
    document.documentElement.removeAttribute('data-popout');
  });

  test('an embedded frame is not a pop-out: no stamp, no client', () => {
    window.history.replaceState({}, '', '?embedded=true&popout=false');
    initPopoutClient.mockClear();

    applyEmbedded();

    expect(document.documentElement.hasAttribute('data-popout')).toBe(false);
    expect(initPopoutClient).not.toHaveBeenCalled();
  });
});

describe('isEmbedded', () => {
  afterEach(() => {
    document.body.className = '';
    window.history.replaceState({}, '', '/');
  });

  test('is true only for ?embedded=true, exactly as applyEmbedded reads it', () => {
    for (const [query, expected] of [
      ['?embedded=true', true],
      ['?embedded=false', false],
      ['?embedded=1', false],
      ['', false],
    ]) {
      window.history.replaceState({}, '', `/${query}`);
      expect(isEmbedded()).toBe(expected);
    }
  });

  test('answers from the URL, so a page can branch on it before applyEmbedded()', () => {
    // The theme role is picked at module load; <body> may not carry the class yet.
    window.history.replaceState({}, '', '/?embedded=true');
    expect(document.body.classList.contains('embedded')).toBe(false);
    expect(isEmbedded()).toBe(true);
  });
});

describe('isPopout', () => {
  afterEach(() => {
    window.history.replaceState({}, '', '/');
  });

  test('is true only for ?popout=true, the rule isEmbedded applies to embedded', () => {
    for (const [query, expected] of [
      ['?popout=true', true],
      ['?popout=true&theme=dark', true],
      ['?popout=false', false],
      ['?popout=1', false],
      ['?embedded=true', false],
      ['', false],
    ]) {
      window.history.replaceState({}, '', `/${query}`);
      expect(isPopout()).toBe(expected);
    }
  });
});

describe('CONTRACT_VERSION', () => {
  test('is a non-empty string', () => {
    expect(typeof CONTRACT_VERSION).toBe('string');
    expect(CONTRACT_VERSION.length).toBeGreaterThan(0);
  });
});

describe('web.terminal: false (data-terminal="off" on <html>)', () => {
  afterEach(() => {
    document.documentElement.removeAttribute('data-terminal');
  });

  test('terminalOffered reads the attribute and defaults to offered', () => {
    expect(terminalOffered()).toBe(true);
    document.documentElement.setAttribute('data-terminal', 'off');
    expect(terminalOffered()).toBe(false);
  });
});
