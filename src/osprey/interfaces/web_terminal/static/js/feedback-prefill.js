// @ts-check
/* OSPREY Web Terminal — Feedback Prefill Builders (pure)
 *
 * Builds the two outbound prefill URLs of the feedback dialog — a GitHub
 * new-issue link and a `mailto:` draft — plus the shared issue body they carry.
 * Deliberately pure: no DOM, no fetch, no module state. The dialog owns the
 * clipboard, the window.open and the paper-trail POST; this module only does
 * string arithmetic, so the caps can be tested without a browser.
 *
 * The cap is on the PERCENT-ENCODED URL, not on the text. Encoding inflates
 * non-ASCII badly — one astral code point (2 UTF-16 units) becomes 12 encoded
 * characters — so there is no fixed ratio to slice by. Each builder therefore
 * binary-searches the raw body for the longest prefix whose *finished* URL fits
 * and measures every candidate for real. Two cuts are illegal and neither can
 * happen here:
 *
 *   - Splitting a percent-escape. Impossible by construction: the body is cut
 *     BEFORE it is encoded, never after.
 *   - Splitting a code point. A UTF-16 slice can land between a surrogate pair,
 *     which both mangles the character and makes `encodeURIComponent` throw.
 *     `sliceCodePoints` snaps such a cut back by one unit, and `sanitize`
 *     repairs lone surrogates that arrived in the input already broken.
 *
 * The title/subject is never shortened — it is the part a maintainer triages
 * from an inbox list. When the title alone busts the cap the body collapses to
 * the marker and the URL is returned over-cap rather than silently mangled.
 */

/** Percent-encoded length cap for the GitHub new-issue URL. */
export const GITHUB_URL_LIMIT = 8000;

/**
 * Percent-encoded length cap for a `mailto:` URL. Well under the ~2000-char
 * floor that the weakest mail handlers and OS URL dispatchers accept.
 */
export const MAILTO_URL_LIMIT = 1800;

/**
 * Marker left in place of the text that did not fit. The dialog auto-copies the
 * unified payload whenever this appears, so the promise it makes holds.
 */
export const TRUNCATION_MARKER = '[truncated — full text on your clipboard]';

/** Heading of the rendered metadata block in a prefilled body. */
const METADATA_HEADING = '### Deployment metadata';

/** Wording of the paste placeholder. Pinned by test — the popover echoes it. */
const CONTEXT_PLACEHOLDER_WORDING = 'paste the copied session context here';

const CONTEXT_PLACEHOLDER = [
  '<details>',
  `<summary>Session context — ${CONTEXT_PLACEHOLDER_WORDING}</summary>`,
  '',
  `_${CONTEXT_PLACEHOLDER_WORDING}_`,
  '',
  '</details>',
].join('\n');

/** A high surrogate with no low after it, or a low surrogate with no high before it. */
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g;

const ENCODER = new TextEncoder();

/**
 * UTF-8 byte length of a string.
 *
 * `String.length` counts UTF-16 units and is wrong for every byte budget in
 * this feature: `utf8Length('é') === 2` while `'é'.length === 1`.
 *
 * @param {string} value
 * @returns {number} number of UTF-8 bytes
 */
export function utf8Length(value) {
  return ENCODER.encode(value).length;
}

/**
 * Replace unpaired surrogates with U+FFFD.
 *
 * A lone surrogate makes `encodeURIComponent` throw `URIError`, which would
 * take down the whole dialog for what is at worst one mangled character.
 *
 * @param {string} value
 * @returns {string}
 */
function sanitize(value) {
  return String(value ?? '').replace(LONE_SURROGATE, '�');
}

/**
 * `value.slice(0, end)` that never cuts a surrogate pair in half.
 *
 * @param {string} value
 * @param {number} end - desired cut index in UTF-16 units
 * @returns {string}
 */
function sliceCodePoints(value, end) {
  let cut = Math.max(0, Math.min(end, value.length));
  if (cut > 0 && cut < value.length) {
    const before = value.charCodeAt(cut - 1);
    const after = value.charCodeAt(cut);
    const splitsPair =
      before >= 0xd800 && before <= 0xdbff && after >= 0xdc00 && after <= 0xdfff;
    if (splitsPair) cut -= 1;
  }
  return value.slice(0, cut);
}

/**
 * Append the truncation marker to a body prefix, on its own paragraph.
 *
 * Trailing whitespace is dropped first so the marker never floats away from
 * the last real line.
 *
 * @param {string} prefix
 * @returns {string}
 */
function withMarker(prefix) {
  const trimmed = prefix.replace(/\s+$/, '');
  return trimmed ? `${trimmed}\n\n${TRUNCATION_MARKER}` : TRUNCATION_MARKER;
}

/**
 * Longest prefix of `body` whose rendered URL fits `limit`.
 *
 * `render` is measured for real on every candidate, so no assumption is made
 * about how much the encoder inflates the input. Candidate length is monotone
 * in the cut index (a longer prefix never encodes shorter), which is what makes
 * the binary search sound.
 *
 * @param {string} body - already sanitized
 * @param {(candidate: string) => string} render - body -> finished URL
 * @param {number} limit - maximum encoded URL length
 * @returns {{body: string, truncated: boolean}}
 */
function fitBody(body, render, limit) {
  if (render(body).length <= limit) return { body, truncated: false };

  // Fallback when not even the bare marker fits: the caller asked for a title
  // that busts the cap on its own, and shortening it is not ours to do.
  let best = TRUNCATION_MARKER;
  let low = 0;
  let high = body.length;
  while (low <= high) {
    const mid = (low + high) >> 1;
    const candidate = withMarker(sliceCodePoints(body, mid));
    if (render(candidate).length <= limit) {
      best = candidate;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return { body: best, truncated: true };
}

/**
 * Build a prefilled GitHub new-issue URL, capped at {@link GITHUB_URL_LIMIT}
 * encoded characters by shortening the body (never the title).
 *
 * @param {string} repo - `owner/name`
 * @param {string} title - issue title, never truncated
 * @param {string} body - issue body, truncated to fit
 * @returns {{url: string, truncated: boolean}} `truncated` is true when the
 *   body was shortened — the caller should make sure the full text is on the
 *   clipboard.
 */
export function buildGitHubIssueUrl(repo, title, body) {
  const path = sanitize(repo)
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/');
  const encodedTitle = encodeURIComponent(sanitize(title));
  /** @param {string} candidate */
  const render = (candidate) =>
    `https://github.com/${path}/issues/new?title=${encodedTitle}&body=${encodeURIComponent(candidate)}`;

  const fitted = fitBody(sanitize(body), render, GITHUB_URL_LIMIT);
  return { url: render(fitted.body), truncated: fitted.truncated };
}

/**
 * Build a prefilled `mailto:` draft, capped at {@link MAILTO_URL_LIMIT}
 * encoded characters by shortening the body (never the subject).
 *
 * @param {string} address - recipient address
 * @param {string} subject - mail subject, never truncated
 * @param {string} body - mail body, truncated to fit
 * @returns {{url: string, needsAutoCopy: boolean}} `needsAutoCopy` is true when
 *   the body was shortened: the dialog must then copy the unified payload for
 *   the user, because the marker promises the full text is on their clipboard.
 */
export function buildMailto(address, subject, body) {
  // `@` is legal unencoded in the mailto to-part and keeps the URL readable in
  // the browser's own "open your mail app?" prompt; everything else is encoded.
  const to = encodeURIComponent(sanitize(address)).replace(/%40/g, '@');
  const encodedSubject = encodeURIComponent(sanitize(subject));
  /** @param {string} candidate */
  const render = (candidate) =>
    `mailto:${to}?subject=${encodedSubject}&body=${encodeURIComponent(candidate)}`;

  const fitted = fitBody(sanitize(body), render, MAILTO_URL_LIMIT);
  return { url: render(fitted.body), needsAutoCopy: fitted.truncated };
}

/**
 * Render the metadata key/value block, or an empty string when there is
 * nothing worth printing.
 *
 * @param {Record<string, unknown>} metadata
 * @returns {string}
 */
function renderMetadata(metadata) {
  const lines = [];
  for (const [key, value] of Object.entries(metadata)) {
    if (value === null || value === undefined) continue;
    const rendered = String(value).trim();
    if (!rendered) continue;
    lines.push(`- **${key}:** ${rendered}`);
  }
  return lines.length === 0 ? '' : [METADATA_HEADING, '', ...lines].join('\n');
}

/**
 * Compose the issue/mail body: the user's report, the deployment metadata
 * block, and a collapsed placeholder for the session context.
 *
 * The placeholder is where the user pastes the clipboard payload. It is always
 * present, even with the context checkbox off, because the marker left by a
 * truncated body points at the clipboard too.
 *
 * @param {string} text - what the user typed
 * @param {Record<string, unknown>} [metadata] - key/value pairs; empty and
 *   nullish values are dropped, and an empty block is omitted entirely
 * @returns {string} the composed body, ready to hand to a URL builder
 */
export function buildPrefillBody(text, metadata) {
  const sections = [];
  const report = String(text ?? '').trim();
  if (report) sections.push(report);
  const block = renderMetadata(metadata ?? {});
  if (block) sections.push(block);
  sections.push(CONTEXT_PLACEHOLDER);
  return sections.join('\n\n');
}
