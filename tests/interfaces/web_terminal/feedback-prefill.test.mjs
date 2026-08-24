// @ts-check
/* Unit tests for the feedback prefill builders (pure functions, no DOM).
 *
 * The load-bearing property is the CAP: what GitHub and mail clients refuse is
 * the *percent-encoded* URL, which inflates non-ASCII input by up to 12x
 * (one astral code point -> 12 encoded chars). Every cap assertion here is
 * therefore made on `result.url.length`, never on the raw body length.
 *
 * The second property is the two-mode contract: a built URL carries either the
 * FULL body or the POINTER line, never a partial slice. A body that does not
 * fit is *replaced*, so there is no truncation arithmetic to get wrong and the
 * draft can never open with half a report that a later paste would duplicate.
 */

import { test, expect, describe } from 'vitest';

import {
  GITHUB_URL_LIMIT,
  MAILTO_URL_LIMIT,
  PASTE_POINTER,
  buildGitHubIssueUrl,
  buildMailto,
  buildPrefillBody,
  utf8Length,
} from '../../../src/osprey/interfaces/web_terminal/static/js/feedback-prefill.js';

/**
 * Read one query parameter back out of a built URL.
 *
 * `decodeURIComponent` throws `URIError` on a split percent-escape, so a
 * successful read is itself a well-formedness assertion.
 *
 * @param {string} url
 * @param {string} name
 * @returns {string} the decoded value
 */
function param(url, name) {
  const match = new RegExp(`[?&]${name}=([^&]*)`).exec(url);
  if (match === null) throw new Error(`no ?${name}= in ${url.slice(0, 80)}`);
  return decodeURIComponent(match[1]);
}

/** A 50 KB ASCII body. */
const BIG_ASCII = 'lorem ipsum dolor sit amet '.repeat(1900); // ~51 KB

/** An emoji-heavy body: every code point is 4 UTF-8 bytes / 12 encoded chars. */
const BIG_EMOJI = '😀🚀'.repeat(3000); // 6000 astral code points

describe('utf8Length', () => {
  test('counts UTF-8 bytes, not UTF-16 units', () => {
    expect(utf8Length('é')).toBe(2);
    expect(utf8Length('')).toBe(0);
    expect(utf8Length('abc')).toBe(3);
    // An astral code point is 2 UTF-16 units but 4 UTF-8 bytes.
    expect('😀'.length).toBe(2);
    expect(utf8Length('😀')).toBe(4);
    expect(utf8Length('—')).toBe(3);
  });

  test('is additive over concatenation', () => {
    expect(utf8Length('é😀abc')).toBe(utf8Length('é') + utf8Length('😀') + 3);
  });
});

describe('buildGitHubIssueUrl', () => {
  test('carries the full body when it fits', () => {
    const result = buildGitHubIssueUrl('als-apg/osprey', 'Feedback', 'hello world');
    expect(result.needsPaste).toBe(false);
    expect(result.url.startsWith('https://github.com/als-apg/osprey/issues/new?')).toBe(true);
    expect(param(result.url, 'title')).toBe('Feedback');
    expect(param(result.url, 'body')).toBe('hello world');
  });

  test('replaces a 50 KB body with the pointer line, under the cap', () => {
    const result = buildGitHubIssueUrl('als-apg/osprey', 'Feedback', BIG_ASCII);
    expect(result.url.length).toBeLessThanOrEqual(GITHUB_URL_LIMIT);
    expect(result.needsPaste).toBe(true);
    expect(param(result.url, 'body')).toBe(PASTE_POINTER);
    expect(param(result.url, 'title')).toBe('Feedback');
  });

  test('an emoji-heavy body flips to the pointer, never a partial slice', () => {
    const result = buildGitHubIssueUrl('als-apg/osprey', 'Feedback', BIG_EMOJI);
    expect(result.url.length).toBeLessThanOrEqual(GITHUB_URL_LIMIT);
    expect(result.needsPaste).toBe(true);
    expect(param(result.url, 'body')).toBe(PASTE_POINTER);
  });

  test('never shortens the title, even when the body has to go', () => {
    const title = `Feedback ${'t'.repeat(300)}`;
    const result = buildGitHubIssueUrl('als-apg/osprey', title, BIG_ASCII);
    expect(param(result.url, 'title')).toBe(title);
    expect(result.url.length).toBeLessThanOrEqual(GITHUB_URL_LIMIT);
  });

  test('a title that alone busts the cap is returned over-cap, not mangled', () => {
    const title = 'x'.repeat(GITHUB_URL_LIMIT * 2);
    const result = buildGitHubIssueUrl('als-apg/osprey', title, 'body text');
    expect(param(result.url, 'title')).toBe(title);
    expect(result.needsPaste).toBe(true);
    expect(param(result.url, 'body')).toBe(PASTE_POINTER);
  });

  test('flips needsPaste exactly at the cap boundary', () => {
    // Grow the body across the cap and assert the flag tracks the pointer
    // swap precisely — never set early, never missed late.
    for (const size of [1, 1000, 5000, 7000, 7900, 8100, 10000]) {
      const result = buildGitHubIssueUrl('als-apg/osprey', 'S', 'x'.repeat(size));
      expect(result.url.length).toBeLessThanOrEqual(GITHUB_URL_LIMIT);
      expect(result.needsPaste).toBe(param(result.url, 'body') === PASTE_POINTER);
      if (!result.needsPaste) expect(param(result.url, 'body')).toBe('x'.repeat(size));
    }
  });

  test('percent-encodes the repo path instead of interpolating it raw', () => {
    const result = buildGitHubIssueUrl('als apg/osprey', 'T', 'b');
    expect(result.url.startsWith('https://github.com/als%20apg/osprey/issues/new?')).toBe(true);
  });

  test('survives a lone surrogate in the body', () => {
    const result = buildGitHubIssueUrl('als-apg/osprey', 'T', `ok\uD800tail`);
    expect(param(result.url, 'body')).toBe('ok�tail');
  });
});

describe('buildMailto', () => {
  test('carries the full body when it fits', () => {
    const result = buildMailto('osprey@example.org', 'OSPREY feedback', 'hello');
    expect(result.needsPaste).toBe(false);
    expect(result.url.startsWith('mailto:osprey@example.org?')).toBe(true);
    expect(param(result.url, 'subject')).toBe('OSPREY feedback');
    expect(param(result.url, 'body')).toBe('hello');
  });

  test('replaces a 50 KB body with the pointer line, under the cap', () => {
    const result = buildMailto('osprey@example.org', 'OSPREY feedback', BIG_ASCII);
    expect(result.url.length).toBeLessThanOrEqual(MAILTO_URL_LIMIT);
    expect(result.needsPaste).toBe(true);
    expect(param(result.url, 'body')).toBe(PASTE_POINTER);
  });

  test('the pointer line is plain text that works in a mail body', () => {
    // Pinned: the same line lands in GitHub markdown and plaintext drafts, so
    // it must carry no markup of either kind.
    expect(PASTE_POINTER).toBe(
      'Your full report is on your clipboard — paste it here, replacing this line.'
    );
    expect(PASTE_POINTER).not.toMatch(/[<>#*_`[\]]/);
  });

  test('flips needsPaste exactly at the cap boundary', () => {
    for (const size of [1, 100, 500, 900, 1100, 1300, 1500, 2000, 5000]) {
      const result = buildMailto('osprey@example.org', 'S', 'x'.repeat(size));
      expect(result.url.length).toBeLessThanOrEqual(MAILTO_URL_LIMIT);
      expect(result.needsPaste).toBe(param(result.url, 'body') === PASTE_POINTER);
    }
  });

  test('keeps the address readable but encodes the rest of it', () => {
    const result = buildMailto('a b@example.org', 'S', 'x');
    expect(result.url.startsWith('mailto:a%20b@example.org?')).toBe(true);
  });
});

describe('buildPrefillBody', () => {
  test('renders the text and the metadata block, nothing else', () => {
    const body = buildPrefillBody('The lattice panel froze.', {
      'OSPREY version': '1.4.0',
      Deployment: 'ALS control room',
    });
    expect(body).toContain('The lattice panel froze.');
    expect(body).toContain('OSPREY version');
    expect(body).toContain('1.4.0');
    expect(body).toContain('ALS control room');
    // No paste placeholder: a FULL-mode draft is complete as opened, and a
    // POINTER-mode draft never contains this body at all.
    expect(body).not.toContain('<details>');
    expect(body).not.toContain('paste');
  });

  test('keeps the report text ahead of the metadata block', () => {
    const body = buildPrefillBody('report text', { Version: '1.0' });
    expect(body.indexOf('report text')).toBeLessThan(body.indexOf('Version'));
  });

  test('omits the metadata block entirely when there is no metadata', () => {
    const body = buildPrefillBody('just text', {});
    expect(body).toBe('just text');

    const noArg = buildPrefillBody('just text');
    expect(noArg).toBe(body);
  });

  test('drops empty metadata values but keeps falsy-but-real ones', () => {
    const body = buildPrefillBody('t', {
      Kept: 0,
      AlsoKept: false,
      Dropped: null,
      AlsoDropped: undefined,
      Blank: '   ',
    });
    expect(body).toContain('Kept');
    expect(body).toContain('AlsoKept');
    expect(body).not.toContain('Dropped');
    expect(body).not.toContain('Blank');
  });

  test('feeds the builders: an oversize composed body lands in pointer mode', () => {
    const composed = buildPrefillBody(BIG_EMOJI, { Version: '1.4.0' });
    const gh = buildGitHubIssueUrl('als-apg/osprey', 'Feedback', composed);
    const mail = buildMailto('osprey@example.org', 'Feedback', composed);
    expect(gh.url.length).toBeLessThanOrEqual(GITHUB_URL_LIMIT);
    expect(mail.url.length).toBeLessThanOrEqual(MAILTO_URL_LIMIT);
    expect(gh.needsPaste).toBe(true);
    expect(mail.needsPaste).toBe(true);
  });
});
