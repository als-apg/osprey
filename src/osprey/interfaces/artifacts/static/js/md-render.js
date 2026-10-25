// @ts-check
/**
 * OSPREY Artifact Gallery — the markdown+KaTeX render pipeline.
 *
 * The single owner of the marked + hljs + KaTeX algorithm: the gallery's
 * preview pane (preview-content.js) and the standalone server-rendered
 * markdown page (app.py's `_MARKDOWN_PAGE_TEMPLATE`, which loads this file
 * as a module) both render through these two functions, so the
 * math-placeholder strategy, the code-highlighting renderer, and the
 * KaTeX error fallback cannot drift between the two surfaces.
 *
 * Stateless aside from `configureMarked`'s one-time idempotency flag.
 * `marked`, `hljs`, `katex` and `DOMPurify` are page globals loaded via
 * classic vendor <script> tags before this module runs; every use is
 * feature-guarded so the pipeline degrades (escaped code, raw math source)
 * rather than throwing when one is absent.
 *
 * Trust boundary: the markdown is agent-written and the gallery runs on the
 * console's origin, so every HTML string this module returns has passed
 * through `DOMPurify.sanitize` (the sanitizer chat-markdown.js uses). The
 * sanitizer runs on the FINAL string, after the KaTeX swap: a math
 * placeholder can land inside an attribute (a link target, say), where
 * splicing KaTeX's quoted markup in after sanitizing would break out of the
 * attribute. Without `DOMPurify` no markup is produced at all — the text
 * comes back escaped, as inert plain text.
 *
 * @module md-render
 */

import { escapeHtml } from "/design-system/js/dom.js";

let _markedConfigured = false;

/**
 * The page's sanitizer, or null when DOMPurify is not loaded.
 * @returns {((html: string) => string) | null}
 */
export function markdownSanitizer() {
  const purify = /** @type {any} */ (globalThis).DOMPurify;
  if (!purify || typeof purify.sanitize !== "function") return null;
  return (html) => String(purify.sanitize(html));
}

/**
 * The text as inert HTML: escaped, whitespace kept.
 * @param {string} text
 * @returns {string}
 */
function plainText(text) {
  return `<pre class="osprey-md-plain">${escapeHtml(text)}</pre>`;
}

/**
 * Register the gfm + code-highlighting marked configuration (idempotent).
 * @returns {void}
 */
export function configureMarked() {
  if (_markedConfigured) return;
  _markedConfigured = true;
  if (typeof marked === "undefined") return;

  const renderer = {
    /** @param {{text: string, lang?: string}} tok */
    code({ text, lang }) {
      const src = text ?? "";
      let highlighted = escapeHtml(src);
      if (typeof hljs !== "undefined" && src) {
        try {
          if (lang && hljs.getLanguage(lang)) {
            highlighted = hljs.highlight(src, { language: lang }).value;
          } else {
            highlighted = hljs.highlightAuto(src).value;
          }
        } catch { /* fallback to escaped */ }
      }
      return `<pre><code class="hljs${lang ? ` language-${lang}` : ""}">${highlighted}</code></pre>`;
    },
  };

  marked.use({ gfm: true, breaks: false, renderer });
}

/**
 * Render LaTeX math in markdown text using KaTeX.
 *
 * Strategy: extract $$...$$ (display) and $...$ (inline) blocks BEFORE
 * marked.js parses the text (to prevent marked from mangling LaTeX
 * special characters like _ and ^).  Replace with placeholders, run
 * marked.parse(), then swap KaTeX-rendered HTML back in, then sanitize the
 * whole string. Returns escaped plain text when the sanitizer (or marked) is
 * missing or throws, so script-bearing markdown never yields live markup.
 * @param {string} text
 * @returns {string}
 */
export function renderMathInMarkdown(text) {
  const sanitize = markdownSanitizer();
  if (!sanitize || typeof marked === "undefined") return plainText(text);
  try {
    return sanitize(renderUnsanitized(text));
  } catch {
    return plainText(text);
  }
}

/**
 * marked + KaTeX output, NOT yet sanitized — only renderMathInMarkdown calls it.
 * @param {string} text
 * @returns {string}
 */
function renderUnsanitized(text) {
  if (typeof katex === "undefined") return marked.parse(text);

  /** @type {{key: string, html: string}[]} */
  const placeholders = [];
  let idx = 0;

  /** @param {string} html */
  function placeholder(html) {
    const key = `\x00MATH${idx++}\x00`;
    placeholders.push({ key, html });
    return key;
  }

  /**
   * @param {string} expr
   * @param {boolean} displayMode
   */
  function renderKatex(expr, displayMode) {
    try {
      return katex.renderToString(expr.trim(), {
        displayMode,
        throwOnError: false,
        strict: false,
      });
    } catch {
      const cls = displayMode ? "katex-error-display" : "katex-error-inline";
      return `<span class="${cls}">${escapeHtml(expr)}</span>`;
    }
  }

  // Pass 1: display math $$...$$ (must come before inline $...$)
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, expr) =>
    placeholder(renderKatex(expr, true))
  );

  // Pass 2: inline math $...$ (not preceded/followed by digit to avoid $5)
  text = text.replace(/(?<!\$)(?<!\d)\$(?!\$)(.+?)(?<!\$)\$(?!\d)/g, (_, expr) =>
    placeholder(renderKatex(expr, false))
  );

  // Run marked on the placeholder-injected text
  let html;
  try {
    html = marked.parse(text);
  } catch {
    html = `<p>${escapeHtml(text)}</p>`;
  }

  // Swap placeholders back in; the caller sanitizes the result as a whole.
  for (const { key, html: mathHtml } of placeholders) {
    html = html.replace(key, mathHtml);
  }

  return html;
}
