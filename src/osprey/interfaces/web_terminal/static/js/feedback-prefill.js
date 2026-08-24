// @ts-check
/* OSPREY Web Terminal — Feedback Prefill Builders (pure)
 *
 * Builds the two outbound prefill URLs of the feedback dialog — a GitHub
 * new-issue link and a `mailto:` draft — plus the issue body they carry.
 * Deliberately pure: no DOM, no fetch, no module state. The dialog owns the
 * clipboard, the window.open and the paper-trail POST; this module only does
 * string arithmetic, so the caps can be tested without a browser.
 *
 * A prefilled body has exactly two modes, and nothing in between:
 *
 *   - FULL: the whole composed body (report text + metadata block) rides the
 *     URL. Chosen when it fits the cap; the draft is complete as opened and no
 *     clipboard is involved.
 *   - POINTER: the body is one line telling the reader to paste, and the whole
 *     report travels on the clipboard instead. Chosen when the full body busts
 *     the cap — and chosen up front by the caller whenever session context is
 *     attached, because context never fits a URL (it is sized against GitHub's
 *     64 KB issue-body limit, not the ~8 KB URL one).
 *
 * There is deliberately no third "as much as fits" mode. A partially prefilled
 * draft that the paste is going to replace anyway is worth nothing, and pasting
 * *under* a partial body duplicates everything the URL did carry.
 *
 * The cap is on the PERCENT-ENCODED URL, not on the text: encoding inflates
 * non-ASCII badly (one astral code point — 2 UTF-16 units — becomes 12 encoded
 * characters), so each builder measures the finished URL for real. The
 * title/subject is never shortened — it is the part a maintainer triages from
 * an inbox list. When the title alone busts the cap the URL is returned
 * over-cap rather than silently mangled.
 */

/** Percent-encoded length cap for the GitHub new-issue URL. */
export const GITHUB_URL_LIMIT = 8000;

/**
 * Percent-encoded length cap for a `mailto:` URL. Well under the ~2000-char
 * floor that the weakest mail handlers and OS URL dispatchers accept.
 */
export const MAILTO_URL_LIMIT = 1800;

/**
 * The POINTER-mode body: the one line a draft carries when the report travels
 * by clipboard. Plain text on purpose — it must read the same in a GitHub
 * issue body and in a plaintext mail draft.
 */
export const PASTE_POINTER =
  'Your full report is on your clipboard — paste it here, replacing this line.';

/** Heading of the rendered metadata block in a FULL-mode body. */
const METADATA_HEADING = '### Deployment metadata';

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
 * Build a prefilled GitHub new-issue URL.
 *
 * @param {string} repo - `owner/name`
 * @param {string} title - issue title, never truncated
 * @param {string} body - the desired issue body
 * @returns {{url: string, needsPaste: boolean}} `needsPaste` is true when the
 *   body did not fit the cap and was replaced with {@link PASTE_POINTER} — the
 *   caller must then make sure the full report is on the clipboard, because
 *   the pointer line promises exactly that.
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

  const full = render(sanitize(body));
  if (full.length <= GITHUB_URL_LIMIT) return { url: full, needsPaste: false };
  return { url: render(PASTE_POINTER), needsPaste: true };
}

/**
 * Build a prefilled `mailto:` draft.
 *
 * @param {string} address - recipient address
 * @param {string} subject - mail subject, never truncated
 * @param {string} body - the desired mail body
 * @returns {{url: string, needsPaste: boolean}} `needsPaste` as in
 *   {@link buildGitHubIssueUrl}: the body collapsed to the pointer line and
 *   the caller owes the clipboard the full report.
 */
export function buildMailto(address, subject, body) {
  // `@` is legal unencoded in the mailto to-part and keeps the URL readable in
  // the browser's own "open your mail app?" prompt; everything else is encoded.
  const to = encodeURIComponent(sanitize(address)).replace(/%40/g, '@');
  const encodedSubject = encodeURIComponent(sanitize(subject));
  /** @param {string} candidate */
  const render = (candidate) =>
    `mailto:${to}?subject=${encodedSubject}&body=${encodeURIComponent(candidate)}`;

  const full = render(sanitize(body));
  if (full.length <= MAILTO_URL_LIMIT) return { url: full, needsPaste: false };
  return { url: render(PASTE_POINTER), needsPaste: true };
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
 * Compose the FULL-mode body: the user's report, then the deployment metadata
 * block. This is the complete draft for a send without session context; a
 * caller attaching context never calls this for the URL — it prefers
 * {@link PASTE_POINTER} outright and lets the clipboard carry everything.
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
  return sections.join('\n\n');
}
