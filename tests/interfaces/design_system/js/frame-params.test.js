/**
 * Unit tests for the design-system micro-frontend query-param contract.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/frame-params.test.js
 *
 * Covers CONTRACT_VERSION, applyEmbedded() (the embedded class), isEmbedded()
 * and terminalOffered().
 *
 * NOTE: frame-params.js is imported by RELATIVE path, not the absolute
 * `/design-system/js/frame-params.js` runtime specifier — matching the
 * sibling suites in this directory (see
 * tests/interfaces/design_system/js/dom.test.js).
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import {
  applyEmbedded,
  terminalOffered,
  isEmbedded,
  CONTRACT_VERSION,
  directVisitHref,
  redirectTopLevelVisit,
} from '../../../../src/osprey/interfaces/design_system/static/js/frame-params.js';
import { isUrlWithinMount } from '../../../../src/osprey/interfaces/web_terminal/static/js/panel-iframe-sync.js';

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

  test('a pop-out address means nothing any more', () => {
    window.history.replaceState({}, '', '?popout=true');

    applyEmbedded();

    expect(document.documentElement.hasAttribute('data-popout')).toBe(false);
    expect(document.body.classList.contains('embedded')).toBe(false);
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

describe('CONTRACT_VERSION', () => {
  test('is a non-empty string', () => {
    expect(typeof CONTRACT_VERSION).toBe('string');
    expect(CONTRACT_VERSION).toBe('2');
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

/**
 * The one `at` contract: `at` is the prefix + the bare path, with a capability
 * segment removed, plus the query and the hash. The hub checks it against the
 * panel's prefixed mount, so a lost prefix is a refused link. Everything here
 * is decoded the way the hub decodes it: `URLSearchParams.get`.
 */
describe('directVisitHref: the /u/<user> at encoding', () => {
  /** @param {string} href */
  const visit = (href) => {
    const u = new URL(href, 'http://hub.test');
    const target = directVisitHref(u);
    if (target === null) return null;
    const out = new URL(target, 'http://hub.test');
    return {
      path: out.pathname,
      open: out.searchParams.get('open'),
      at: out.searchParams.get('at'),
      keys: [...out.searchParams.keys()],
      hash: out.hash,
    };
  };

  test('a per-user mount keeps its prefix on the hub and inside at', () => {
    expect(visit('/u/alice/panel/ariel/')).toEqual({
      path: '/u/alice/',
      open: 'ariel',
      at: '/u/alice/panel/ariel/',
      keys: ['open', 'at'],
      hash: '',
    });
  });

  test('a capability segment leaves at, and the prefix stays', () => {
    expect(visit('/u/alice/panel/ariel/~c/TOKEN/log/view')?.at).toBe(
      '/u/alice/panel/ariel/log/view'
    );
    expect(visit('/u/alice/panel-static/mine/~c/TOKEN/index.html')?.at).toBe(
      '/u/alice/panel-static/mine/index.html'
    );
  });

  test('only the segment right after the mount is a capability', () => {
    expect(visit('/u/alice/panel/ariel/docs/~c/keep/page')?.at).toBe(
      '/u/alice/panel/ariel/docs/~c/keep/page'
    );
  });

  test('the query and the hash travel inside at, as one parameter', () => {
    const got = visit('/u/alice/panel/ariel/?q=a%26b+c&x=1#entry?id=X&y=2');
    expect(got?.at).toBe('/u/alice/panel/ariel/?q=a%26b+c&x=1#entry?id=X&y=2');
    expect(got?.keys).toEqual(['open', 'at']);
    expect(got?.hash).toBe('');
  });

  test('a percent-encoded path comes back byte for byte', () => {
    expect(visit('/u/alice/panel/jupyter/doc/tree/my%20note%2Bbook.ipynb')?.at).toBe(
      '/u/alice/panel/jupyter/doc/tree/my%20note%2Bbook.ipynb'
    );
  });

  test('an escaped panel id names the same panel, not an escape of it', () => {
    expect(visit('/u/alice/panel/my%20panel/')?.open).toBe('my panel');
  });

  test('a user called "panel" is a user, not a mount', () => {
    const got = visit('/u/panel/panel/ariel/');
    expect(got?.path).toBe('/u/panel/');
    expect(got?.open).toBe('ariel');
    expect(got?.at).toBe('/u/panel/panel/ariel/');
    expect(visit('/u/panel/static/session.html')).toBeNull();
  });

  test('at stays inside the mount the hub checks it against', () => {
    for (const [href, mount] of [
      ['/u/alice/panel/ariel/#entry?id=X', '/u/alice/panel/ariel/'],
      ['/u/alice/panel/ariel/~c/TOKEN/log', '/u/alice/panel/ariel/'],
      ['/u/alice/panel-static/mine/index.html?x=1', '/u/alice/panel-static/mine/'],
      ['/panel/ariel/#entry?id=X', '/panel/ariel/'],
      ['/panel/ariel/~c/TOKEN/log', '/panel/ariel/'],
    ]) {
      expect(isUrlWithinMount(visit(href)?.at, mount)).toBe(true);
    }
    // A prefixed mount refuses the bare form: the prefix is load-bearing.
    expect(isUrlWithinMount('/panel/ariel/', '/u/alice/panel/ariel/')).toBe(false);
  });

  test('the address the bundle route answers with passes the same check', () => {
    // The Location header test_panel_direct_visit_redirect.py pins, verbatim.
    const location = '/u/alice/?open=mine&at=%2Fu%2Falice%2Fpanel-static%2Fmine%2F%3Fx%3D1';
    const params = new URL(location, 'http://hub.test').searchParams;
    expect(params.get('open')).toBe('mine');
    expect(params.get('at')).toBe('/u/alice/panel-static/mine/?x=1');
    expect(isUrlWithinMount(params.get('at'), '/u/alice/panel-static/mine/')).toBe(true);
  });

  test('without a prefix the hub is the root', () => {
    expect(visit('/panel/ariel/')?.path).toBe('/');
    expect(visit('/panel/ariel/')?.at).toBe('/panel/ariel/');
  });

  test('off a panel mount there is nothing to redirect', () => {
    for (const href of ['/', '/static/session.html', '/u/alice/', '/u/alice/static/x.html']) {
      expect(visit(href)).toBeNull();
    }
    expect(visit('/panel/ariel')).toBeNull();
    expect(visit('/foo/panel/ariel/')).toBeNull();
  });
});

describe('redirectTopLevelVisit', () => {
  /** @param {string} href @param {{framed?: boolean}} [options] */
  const fakeWindow = (href, { framed = false } = {}) => {
    const u = new URL(href, 'http://hub.test');
    /** @type {any} */
    const win = {
      location: { pathname: u.pathname, search: u.search, hash: u.hash, replace: vi.fn() },
    };
    win.top = framed ? {} : win;
    return win;
  };

  test('a top-level visit is replaced by the hub, keeping the hash', () => {
    const win = fakeWindow('/u/alice/panel/ariel/#entry?id=X');
    expect(redirectTopLevelVisit(win)).toBe(true);
    expect(win.location.replace).toHaveBeenCalledWith(
      `/u/alice/?open=ariel&at=${encodeURIComponent('/u/alice/panel/ariel/#entry?id=X')}`
    );
  });

  test.each([
    ['framed', fakeWindow('/panel/ariel/', { framed: true })],
    ['embedded=true', fakeWindow('/panel/ariel/?embedded=true')],
    ['off a mount', fakeWindow('/static/session.html')],
  ])('leaves the page alone when %s', (_name, win) => {
    expect(redirectTopLevelVisit(win)).toBe(false);
    expect(win.location.replace).not.toHaveBeenCalled();
  });

  test('embedded=false is a top-level visit like any other', () => {
    const win = fakeWindow('/panel/ariel/?embedded=false');
    expect(redirectTopLevelVisit(win)).toBe(true);
  });

  test('applyEmbedded runs the guard first', () => {
    const replace = vi.spyOn(window.location, 'replace').mockImplementation(() => {});
    window.history.replaceState({}, '', '/panel/ariel/');
    try {
      applyEmbedded();
      expect(replace).toHaveBeenCalledTimes(1);
      expect(document.body.classList.contains('embedded')).toBe(false);
    } finally {
      replace.mockRestore();
      window.history.replaceState({}, '', '/');
    }
  });
});
