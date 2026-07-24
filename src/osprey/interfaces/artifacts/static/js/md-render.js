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
 * `marked`, `hljs`, and `katex` are page globals loaded via classic
 * vendor <script> tags before this module runs; every use is
 * feature-guarded so the pipeline degrades (plain markdown, escaped code,
 * raw math source) rather than throwing when one is absent.
 *
 * @module md-render
 */

import { escapeHtml } from "/design-system/js/dom.js";

let _markedConfigured = false;

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
 * marked.parse(), then swap KaTeX-rendered HTML back in.
 * @param {string} text
 * @returns {string}
 */
export function renderMathInMarkdown(text) {
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

  // Swap placeholders back in (safe: KaTeX output is sanitized by the library)
  for (const { key, html: mathHtml } of placeholders) {
    html = html.replace(key, mathHtml);
  }

  return html;
}
