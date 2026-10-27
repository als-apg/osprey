// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode chat: markdown and math rendering.
 *
 * {@link renderMarkdownInto} is the one path from model markdown to the DOM.
 * `marked` parses the text, DOMPurify sanitises the HTML, and only the
 * sanitised string reaches `innerHTML`. Math is typeset with KaTeX afterwards,
 * on the DOM, so KaTeX output never passes through the HTML string.
 *
 * The libraries are vendored classic-script globals. They are read off
 * `globalThis` behind an `any` cast, and every use is guarded: when one is
 * missing, the text is written as inert `textContent` instead.
 *
 * @module chat-markdown
 */

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

/** Fenced blocks and code spans; split on it, odd parts are code and stay literal. */
const CODE_SPLIT = /(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]+`)/;

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
  return text
    .split(CODE_SPLIT)
    .map((part, i) => {
      if (i % 2 === 1) return part;
      for (const [re, display] of MATH_PATTERNS) {
        part = part.replace(re, (raw, expr) => {
          found.push({ expr, display, raw });
          return `OSPREYMATH${found.length - 1}Z`;
        });
      }
      return part;
    })
    .join('');
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
 * @param {{ math?: boolean }} [options] - `math` typesets `$…$`, `$$…$$`,
 *   `\(…\)` and `\[…\]` with KaTeX when it is loaded
 * @returns {void}
 */
export function renderMarkdownInto(el, text, { math = false } = {}) {
  const { DOMPurify, hljs, katex } = markdownGlobals();
  const parse = chatMarkedParse();

  // Fail safe: without a parser or a sanitiser we cannot emit HTML, so degrade
  // to text.
  if (!parse || !DOMPurify || typeof DOMPurify.sanitize !== 'function') {
    el.textContent = text;
    return;
  }

  /** @type {Array<{expr: string, display: boolean, raw: string}>} */
  const found = [];
  const withMath = math && katex && typeof katex.render === 'function';
  let clean;
  try {
    clean = DOMPurify.sanitize(parse(withMath ? extractMath(text, found) : text));
  } catch {
    // Any failure in parse/sanitise falls back to inert text rather than
    // risking unsanitised markup reaching the DOM.
    el.textContent = text;
    return;
  }
  el.innerHTML = clean;
  if (found.length) mountMath(el, found, katex);

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
