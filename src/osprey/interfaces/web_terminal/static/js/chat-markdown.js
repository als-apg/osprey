// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode chat: markdown and math rendering.
 *
 * {@link renderMarkdownInto} is the one path from model markdown to the DOM.
 * `marked` parses the text, DOMPurify sanitises the HTML, and only the
 * sanitised string reaches `innerHTML`. The reply is untrusted, so a reply
 * may not make the browser fetch from another site as it renders, and may
 * not paint over the page ({@link sanitizeReply}). Math is typeset with KaTeX afterwards,
 * on the DOM, so KaTeX output never passes through the HTML string. A ```svg
 * block is drawn afterwards too, through chat-svg.js's own sanitising pass.
 *
 * The libraries are vendored classic-script globals. They are read off
 * `globalThis` behind an `any` cast, and every use is guarded: when one is
 * missing, the text is written as inert `textContent` instead.
 *
 * @module chat-markdown
 */

import { createSvgPlaceholder, mountSvgCallouts, openSvgFenceAt } from './chat-svg.js';

/**
 * @returns {{ DOMPurify: any, hljs: any, katex: any }} the vendored
 *   sanitiser/highlighter/math globals, each possibly `undefined`. (`marked` is
 *   reached via {@link chatMarkedParse}, which isolates chat from the shared
 *   singleton.)
 */
function markdownGlobals() {
  const g = /** @type {any} */ (globalThis);
  return { DOMPurify: g.DOMPurify, hljs: g.hljs, katex: g.katex };
}

// ---- Sanitising ---- //

/**
 * Elements a reply may not hold: the ones that take input or send it, and
 * style sheets. The sanitiser also uses the HTML profile only, so there is no
 * inline SVG or MathML (KaTeX is mounted after sanitising, so it needs neither).
 */
const FORBID_TAGS = Object.freeze(['style', 'form', 'input', 'button', 'select', 'textarea']);

/**
 * `style` and `background` paint over the page or load from it. `srcset` lists
 * URLs, so it is dropped rather than checked.
 */
const FORBID_ATTR = Object.freeze(['style', 'background', 'srcset']);

/** The elements that load media, and the attributes of theirs that hold a URL. */
const MEDIA_URL_ATTRS = Object.freeze({
  img: ['src'],
  source: ['src'],
  video: ['src', 'poster'],
  audio: ['src'],
  track: ['src'],
});

/**
 * Whether *value* loads from the page's own origin. A relative URL does; a
 * `data:` URL, another site, and a value that is not a URL do not.
 * @param {string} value
 * @returns {boolean}
 */
function isOwnOrigin(value) {
  try {
    return new URL(value, location.href).origin === location.origin;
  } catch {
    return false;
  }
}

/**
 * DOMPurify hook: remove a media element whose URL leaves the page's origin.
 * @param {Element} node
 */
function dropForeignMedia(node) {
  const attrs = /** @type {Record<string, string[]>} */ (MEDIA_URL_ATTRS)[node.localName ?? ''];
  if (!attrs) return;
  if (
    attrs.some((name) => node.hasAttribute(name) && !isOwnOrigin(node.getAttribute(name) ?? ''))
  ) {
    node.remove();
  }
}

/**
 * Sanitise a reply's HTML. The media hook is on DOMPurify only for this call,
 * so it never reaches chat-svg.js, which sanitises with the same instance.
 * @param {any} DOMPurify
 * @param {string} html - marked's output
 * @returns {string}
 */
function sanitizeReply(DOMPurify, html) {
  const hooks =
    typeof DOMPurify.addHook === 'function' && typeof DOMPurify.removeHook === 'function';
  if (hooks) DOMPurify.addHook('afterSanitizeAttributes', dropForeignMedia);
  try {
    return DOMPurify.sanitize(html, {
      USE_PROFILES: { html: true },
      FORBID_TAGS: [...FORBID_TAGS],
      FORBID_ATTR: [...FORBID_ATTR],
    });
  } finally {
    if (hooks) DOMPurify.removeHook('afterSanitizeAttributes', dropForeignMedia);
  }
}

// ---- Math ---- //

/**
 * Math delimiters, in extraction order, each with whether it is display math.
 * Inline `$…$` follows the pandoc rule — no space just inside either dollar,
 * no digit just outside — so prices like "$5 and $10" stay text; `\$` is a
 * literal dollar.
 * @type {Array<[RegExp, boolean]>}
 */
const MATH_PATTERNS = [
  [/\$\$([\s\S]+?)\$\$/g, true],
  [/\\\[([\s\S]+?)\\\]/g, true],
  [/\\\(([\s\S]+?)\\\)/g, false],
  [/(?<![$\d\\])\$(?![\s$])([^\n$]+?)(?<![\s\\])\$(?!\d)/g, false],
];

/**
 * The code in a reply, which math must leave alone: a fenced block (it runs to
 * its closing fence, which is at least as long as the opening one, or to the
 * end of the text while the fence is still open), a multi-backtick span, and
 * a single-line code span. A backtick fence's info string holds no backtick,
 * so "```x```" is a span, as in CommonMark.
 */
const CODE_RANGE =
  /^[ \t]*(`{3,}(?![^\n]*`)|~{3,})[^\n]*(?:\n[\s\S]*?)?(?:\n[ \t]*\1[`~]*[ \t]*(?=\n|$)|(?![\s\S]))|(`{2,})(?!`)[\s\S]*?(?<!`)\2(?!`)|`[^`\n]+`/gm;

/** The placeholder a math expression rides through marked and DOMPurify as. */
const MATH_TOKEN = /OSPREYMATH(\d+)Z/g;

/**
 * Swap every math expression outside code for a plain-text placeholder, so
 * marked never sees (and mangles) its `_`, `^` and `\`.
 * @param {string} text
 * @param {Array<{expr: string, display: boolean, raw: string}>} found - filled in
 * @returns {string}
 */
function extractMath(text, found) {
  /** @param {string} prose */
  const swap = (prose) => {
    for (const [re, display] of MATH_PATTERNS) {
      prose = prose.replace(re, (raw, expr) => {
        found.push({ expr, display, raw });
        return `OSPREYMATH${found.length - 1}Z`;
      });
    }
    return prose;
  };
  let out = '';
  let last = 0;
  for (const m of text.matchAll(CODE_RANGE)) {
    out += swap(text.slice(last, m.index)) + m[0];
    last = (m.index ?? 0) + m[0].length;
  }
  return out + swap(text.slice(last));
}

/**
 * Typeset the placeholders left in `el`'s text nodes. Done on the DOM after
 * sanitising, so KaTeX output never passes through, or around, the HTML string.
 * @param {HTMLElement} el
 * @param {Array<{expr: string, display: boolean, raw: string}>} found
 * @param {any} katex
 */
function mountMath(el, found, katex) {
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  /** @type {Text[]} */
  const nodes = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    if (n.nodeValue?.includes('OSPREYMATH')) nodes.push(/** @type {Text} */ (n));
  }
  for (const node of nodes) {
    const s = node.nodeValue ?? '';
    // Code is never typeset (marked can make code of text the extraction took
    // for math, such as an indented block): it gets its own characters back.
    if (node.parentElement?.closest('pre, code')) {
      node.nodeValue = s.replace(MATH_TOKEN, (token, i) => found[Number(i)]?.raw ?? token);
      continue;
    }
    const frag = document.createDocumentFragment();
    let last = 0;
    for (const m of s.matchAll(MATH_TOKEN)) {
      frag.append(s.slice(last, m.index));
      const item = found[Number(m[1])];
      const span = document.createElement('span');
      try {
        katex.render(item.expr.trim(), span, {
          displayMode: item.display,
          throwOnError: false,
          strict: false,
        });
      } catch {
        span.textContent = item ? item.raw : m[0];
      }
      frag.append(span);
      last = (m.index ?? 0) + m[0].length;
    }
    frag.append(s.slice(last));
    node.replaceWith(frag);
  }
}

/**
 * Put each table in a box that scrolls sideways, so a wide table keeps its
 * natural width instead of being squeezed into the bubble. Done on the DOM
 * after sanitising, so the box is not subject to the sanitiser's rules.
 * @param {HTMLElement} el
 */
function wrapTables(el) {
  for (const table of el.querySelectorAll('table')) {
    const box = document.createElement('div');
    box.className = 'osprey-md-table-scroll';
    table.replaceWith(box);
    box.append(table);
  }
}

/**
 * Identity of the global `marked` the memoised {@link chatMarkedParse} was
 * last built against, and the parse function derived from it. Kept module-level
 * so the (stable) production singleton yields exactly one isolated instance,
 * while per-test global swaps — which change the identity — force a rebuild.
 * @type {any}
 */
let lastMarked = null;
/** @type {((text: string) => string) | null} */
let chatParse = null;

/**
 * A `parse(markdown) -> html` function isolated from the shared `marked`
 * singleton, or `null` when `marked` is absent.
 *
 * The global `marked` is a page-wide singleton that any script may
 * reconfigure with `marked.use(...)`. marked v12 exposes its `Marked` class
 * on the UMD global, so this module builds its own instance with a plain
 * config and parses through that, immune to whatever the rest of the page
 * does to the singleton. When the constructor is absent (an older or partial
 * vendored build) it falls back to the global `marked.parse`.
 *
 * Resolved lazily rather than at module load: the vendored global loads before
 * this module in the browser, but tests stub `globalThis.marked` per case, so
 * the instance is (re)built whenever the observed global changes identity.
 *
 * @returns {((text: string) => string) | null}
 */
function chatMarkedParse() {
  const marked = /** @type {any} */ (globalThis).marked;
  if (marked === lastMarked) return chatParse;
  lastMarked = marked;

  if (!marked) {
    chatParse = null;
  } else if (typeof marked.Marked === 'function') {
    try {
      const instance = new marked.Marked({ gfm: true, breaks: false });
      chatParse = (text) => instance.parse(text);
    } catch {
      chatParse = typeof marked.parse === 'function' ? (text) => marked.parse(text) : null;
    }
  } else {
    chatParse = typeof marked.parse === 'function' ? (text) => marked.parse(text) : null;
  }
  return chatParse;
}

/**
 * Render markdown text into `el` as sanitised HTML, then syntax-highlight any
 * code blocks. This is the web terminal's one trust boundary for markdown:
 * the chat log and the scaffold Preview both render through it.
 *
 * HTML is produced only when BOTH `marked` and `DOMPurify` are available: the
 * marked output is untrusted HTML, so it must never reach `innerHTML` without
 * passing through `DOMPurify.sanitize` first. If either library is missing (or
 * either step throws), the text is written as inert `textContent` — never as
 * markup. `hljs` is a purely additive enhancement and is skipped when absent.
 *
 * @param {HTMLElement} el - target element; its contents are replaced
 * @param {string} text - raw markdown (model output)
 * @param {{ math?: boolean, svg?: boolean }} [options] - `math` typesets
 *   `$…$`, `$$…$$`, `\(…\)` and `\[…\]` with KaTeX when it is loaded; `svg`
 *   draws each complete ```svg block (see chat-svg.js)
 * @returns {void}
 */
export function renderMarkdownInto(el, text, { math = false, svg = false } = {}) {
  const { DOMPurify, hljs, katex } = markdownGlobals();
  const parse = chatMarkedParse();

  // Fail safe: without a parser or a sanitiser we cannot emit HTML, so degrade
  // to text.
  if (!parse || !DOMPurify || typeof DOMPurify.sanitize !== 'function') {
    el.textContent = text;
    return;
  }

  // A drawing still streaming is not shown as the code it is made of: the
  // reply is drawn up to its fence, and a placeholder stands in until it closes.
  const pendingAt = svg ? openSvgFenceAt(text) : -1;
  const source = pendingAt >= 0 ? text.slice(0, pendingAt) : text;

  /** @type {Array<{expr: string, display: boolean, raw: string}>} */
  const found = [];
  const withMath = math && katex && typeof katex.render === 'function';
  let clean;
  try {
    clean = sanitizeReply(DOMPurify, parse(withMath ? extractMath(source, found) : source));
  } catch {
    // Any failure in parse/sanitise falls back to inert text rather than
    // risking unsanitised markup reaching the DOM.
    el.textContent = text;
    return;
  }
  el.innerHTML = clean;
  wrapTables(el);
  if (found.length) mountMath(el, found, katex);
  if (pendingAt >= 0) el.append(createSvgPlaceholder());
  if (svg) {
    try {
      mountSvgCallouts(el, DOMPurify);
    } catch {
      // A drawing that fails stays the code block it was.
    }
  }

  if (hljs && typeof hljs.highlightElement === 'function') {
    el.querySelectorAll('pre code').forEach((block) => {
      try {
        hljs.highlightElement(/** @type {HTMLElement} */ (block));
      } catch {
        // A single block failing to highlight must not break the message.
      }
    });
  }
}
