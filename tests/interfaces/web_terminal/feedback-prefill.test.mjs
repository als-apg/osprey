// @ts-check
/* Unit tests for the feedback prefill builders (pure functions, no DOM).
 *
 * The load-bearing property is the CAP: what GitHub and mail clients refuse is
 * the *percent-encoded* URL, which inflates non-ASCII input by up to 12x
 * (one astral code point -> 12 encoded chars). Every cap assertion here is
 * therefore made on `result.url.length`, never on the raw body length.
 *
 * The second property is that a truncated body stays *decodable*: cutting the
 * encoded string would routinely land inside a `%E2` escape, and cutting the
 * raw string at an arbitrary UTF-16 index would leave a lone surrogate that
 * makes `encodeURIComponent` throw. Both are asserted by round-tripping every
 * built URL through `decodeURIComponent`, which throws on a split escape.
 */

import { test, expect, describe } from 'vitest';

import {
  GITHUB_URL_LIMIT,
  MAILTO_URL_LIMIT,
  TRUNCATION_MARKER,
  buildGitHubIssueUrl,
  buildMailto,
  buildPrefillBody,
  utf8Length,
} from '../../../src/osprey/interfaces/web_terminal/static/js/feedback-prefill.js';

const CONTEXT_PLACEHOLDER_WORDING = 'paste the copied session context here';

/**
 * Read one query parameter back out of a built URL.
 *
 * `decodeURIComponent` throws `URIError` on a truncated percent-escape, so a
 * successful read is itself the "never split an escape" assertion.
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
  test('builds the new-issue URL untruncated when it fits', () => {
    const result = buildGitHubIssueUrl('als-apg/osprey', 'Feedback', 'hello world');
    expect(result.truncated).toBe(false);
    expect(result.url.startsWith('https://github.com/als-apg/osprey/issues/new?')).toBe(true);
    expect(param(result.url, 'title')).toBe('Feedback');
    expect(param(result.url, 'body')).toBe('hello world');
    expect(result.url).not.toContain(TRUNCATION_MARKER);
  });

  test('holds the encoded URL under the cap for a 50 KB body', () => {
    const result = buildGitHubIssueUrl('als-apg/osprey', 'Feedback', BIG_ASCII);
    expect(result.url.length).toBeLessThanOrEqual(GITHUB_URL_LIMIT);
    expect(result.truncated).toBe(true);
    expect(param(result.url, 'body').endsWith(TRUNCATION_MARKER)).toBe(true);
    expect(param(result.url, 'title')).toBe('Feedback');
  });

  test('holds the encoded URL under the cap for an emoji-heavy body', () => {
    const result = buildGitHubIssueUrl('als-apg/osprey', 'Feedback', BIG_EMOJI);
    expect(result.url.length).toBeLessThanOrEqual(GITHUB_URL_LIMIT);
    expect(result.truncated).toBe(true);

    // Round-trips (no split escape) and no code point was cut in half: a
    // severed surrogate pair would decode to U+FFFD.
    const decoded = param(result.url, 'body');
    expect(decoded).not.toContain('�');
    expect(decoded.endsWith(TRUNCATION_MARKER)).toBe(true);
    expect(decoded.startsWith('😀')).toBe(true);
  });

  test('truncates the body, never the title', () => {
    const title = `Feedback ${'t'.repeat(300)}`;
    const result = buildGitHubIssueUrl('als-apg/osprey', title, BIG_ASCII);
    expect(param(result.url, 'title')).toBe(title);
    expect(result.url.length).toBeLessThanOrEqual(GITHUB_URL_LIMIT);
  });

  test('reports truncation rather than trimming a title that alone busts the cap', () => {
    const title = 'x'.repeat(GITHUB_URL_LIMIT * 2);
    const result = buildGitHubIssueUrl('als-apg/osprey', title, 'body text');
    expect(param(result.url, 'title')).toBe(title);
    expect(result.truncated).toBe(true);
    // The body is reduced to the marker alone — nothing more can be given back.
    expect(param(result.url, 'body')).toBe(TRUNCATION_MARKER);
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
  test('builds the draft untruncated when it fits', () => {
    const result = buildMailto('osprey@example.org', 'OSPREY feedback', 'hello');
    expect(result.needsAutoCopy).toBe(false);
    expect(result.url.startsWith('mailto:osprey@example.org?')).toBe(true);
    expect(param(result.url, 'subject')).toBe('OSPREY feedback');
    expect(param(result.url, 'body')).toBe('hello');
  });

  test('holds the encoded URL under the cap for a 50 KB body', () => {
    const result = buildMailto('osprey@example.org', 'OSPREY feedback', BIG_ASCII);
    expect(result.url.length).toBeLessThanOrEqual(MAILTO_URL_LIMIT);
    expect(result.needsAutoCopy).toBe(true);
    expect(param(result.url, 'body').endsWith(TRUNCATION_MARKER)).toBe(true);
  });

  test('holds the encoded URL under the cap for an emoji-heavy body', () => {
    const result = buildMailto('osprey@example.org', 'OSPREY feedback', BIG_EMOJI);
    expect(result.url.length).toBeLessThanOrEqual(MAILTO_URL_LIMIT);
    const decoded = param(result.url, 'body');
    expect(decoded).not.toContain('�');
    expect(decoded.endsWith(TRUNCATION_MARKER)).toBe(true);
  });

  test('uses the exact truncation marker from the spec', () => {
    expect(TRUNCATION_MARKER).toBe('[truncated — full text on your clipboard]');
    const result = buildMailto('osprey@example.org', 'S', BIG_ASCII);
    expect(param(result.url, 'body')).toContain('[truncated — full text on your clipboard]');
  });

  test('flips needsAutoCopy only when truncation happened', () => {
    // Grow the body one step at a time across the cap boundary and assert the
    // flag tracks the marker exactly — never set early, never missed late.
    for (const size of [1, 100, 500, 900, 1100, 1300, 1500, 2000, 5000]) {
      const result = buildMailto('osprey@example.org', 'S', 'x'.repeat(size));
      expect(result.url.length).toBeLessThanOrEqual(MAILTO_URL_LIMIT);
      expect(result.needsAutoCopy).toBe(param(result.url, 'body').includes(TRUNCATION_MARKER));
    }
  });

  test('keeps the address readable but encodes the rest of it', () => {
    const result = buildMailto('a b@example.org', 'S', 'x');
    expect(result.url.startsWith('mailto:a%20b@example.org?')).toBe(true);
  });
});

describe('buildPrefillBody', () => {
  test('renders text, metadata and the paste placeholder', () => {
    const body = buildPrefillBody('The lattice panel froze.', {
      'OSPREY version': '1.4.0',
      Deployment: 'ALS control room',
    });
    expect(body).toContain('The lattice panel froze.');
    expect(body).toContain('OSPREY version');
    expect(body).toContain('1.4.0');
    expect(body).toContain('ALS control room');
    expect(body).toContain('<details>');
    expect(body).toContain('</details>');
    expect(body).toContain('<summary>');
    expect(body).toContain(CONTEXT_PLACEHOLDER_WORDING);
  });

  test('keeps the report text ahead of the metadata block', () => {
    const body = buildPrefillBody('report text', { Version: '1.0' });
    expect(body.indexOf('report text')).toBeLessThan(body.indexOf('Version'));
    expect(body.indexOf('Version')).toBeLessThan(body.indexOf('<details>'));
  });

  test('omits the metadata block entirely when there is no metadata', () => {
    const body = buildPrefillBody('just text', {});
    expect(body).not.toContain('###');
    expect(body).toContain('just text');
    expect(body).toContain(CONTEXT_PLACEHOLDER_WORDING);

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

  test('feeds the builders: the composed body still lands under both caps', () => {
    const composed = buildPrefillBody(BIG_EMOJI, { Version: '1.4.0' });
    const gh = buildGitHubIssueUrl('als-apg/osprey', 'Feedback', composed);
    const mail = buildMailto('osprey@example.org', 'Feedback', composed);
    expect(gh.url.length).toBeLessThanOrEqual(GITHUB_URL_LIMIT);
    expect(mail.url.length).toBeLessThanOrEqual(MAILTO_URL_LIMIT);
    expect(gh.truncated).toBe(true);
    expect(mail.needsAutoCopy).toBe(true);
  });
});
