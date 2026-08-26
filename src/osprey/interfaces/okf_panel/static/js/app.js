/*
 * OKF Knowledge Panel — read-only SPA shell.
 *
 * PROXY PATH DECISION
 * -------------------
 * This panel is served standalone at `/` (local/dev) AND behind osprey's
 * web-terminal reverse proxy at `/panel/okf/` (production).
 *
 * osprey's proxy (web_terminal/routes/proxy.py) rewrites root-absolute paths
 * NOT ONLY in served HTML but ALSO inside JS/CSS string literals: see
 * `_rewrite_content`, which runs `re.sub(r'(?<=["'`])' + prefix, ...)` over
 * any response whose content-type is text/html, (text|application)/javascript,
 * or text/css. The `_REWRITE_PREFIXES` tuple includes `/static/` and `/api/`
 * (plus a bare `/api`). So a literal like fetch("/api/concept") served from
 * /static/js/app.js becomes fetch("/panel/okf/api/concept") in the browser.
 *
 * Therefore we use PLAIN ROOT-ABSOLUTE paths everywhere (both HTML asset refs
 * and JS fetch() literals). No runtime base-prefix derivation is needed — the
 * proxy handles the prefixing, and at `/` the paths are already correct.
 *
 * Constraint imposed by the rewrite regex: each rewritten path must appear as
 * a string literal *beginning immediately after the opening quote* (the regex
 * uses a quote lookbehind). So we always write the path as its own leading
 * literal, e.g. fetch("/api/concept?id=" + encodeURIComponent(id)), never
 * building the "/api" segment dynamically. The vendor file itself is excluded
 * from rewriting by the proxy (`/vendor/` in path), which is harmless: it
 * contains no /static or /api literals we rely on, and its <script src> in
 * index.html is rewritten as part of the HTML response.
 */

import { initTheme } from "/design-system/js/theme-manager.js";
import { applyEmbedded, isEmbedded } from "/design-system/js/frame-params.js";
import { debounce } from "/design-system/js/dom.js";
import {
  contributeHeader,
  onHeaderAction,
  isSimpleMode,
} from "/design-system/js/header-contrib.js";
import "/design-system/js/components/osprey-display-menu.js";
import { el, isFallback, readPanelParams, STRUCTURE_MARKER } from "./helpers.js";
import { initTree, renderTree, highlightActive, highlightStructure, selectConcept } from "./tree.js";
import {
  initSearchResults,
  clearSearchResults,
  renderSearchMessage,
  renderSearchResults,
} from "./search.js";
import { initSidebarResize } from "./resize.js";

// Standalone, this page owns its own theme chrome (the header
// <osprey-display-menu>), so it runs theme-manager.js in the hub role:
// persistence, OS auto-follow and ?theme= handling all come with it, and
// broadcast is a structural no-op on a page with no iframes. Embedded in the
// Web Terminal hub it is a follower instead: theme-boot.js already applied
// data-theme pre-paint, and this attaches the postMessage listener for the
// hub's live broadcasts.
initTheme({ role: isEmbedded() ? "follower" : "hub" });

/**
 * Publish this panel's tile-bar contribution. Embedded in Expert, the sidebar's
 * search box lives in the hub's tile bar instead (style.css hides the in-body
 * one), rendered as the hub's own magnifier pill.
 *
 * Simple contributes nothing: the hub collapses a service tile's bar to zero
 * height there, so a contributed box would be invisible — and this panel's
 * Simple layout is a reading view that drops the whole sidebar anyway.
 *
 * Sent WHOLE (the hub replaces, never diffs) and a no-op standalone.
 */
function publishHeaderContribution() {
  contributeHeader(
    isSimpleMode()
      ? []
      : [
          // Placeholder matches the in-body input in index.html.
          { kind: "search", id: "search", placeholder: "Search concepts…" },
        ]
  );
}

// Live Expert<->Simple switch broadcast by the hub (same-origin postMessage),
// the mode-axis sibling of the osprey-theme-change follower wired by initTheme
// above. mode-boot.js already stamped the initial data-ui-mode pre-paint; this
// is the runtime flip. The Simple layout is pure CSS gated on the attribute
// (see style.css), so stamping <html> covers the body; the tile-bar
// contribution is mode-dependent, so re-publish it too — that is what moves the
// search box between the bar and the body on a live flip.
window.addEventListener("message", (e) => {
  if (e.origin !== window.location.origin) return;
  if (e.data && e.data.type === "osprey-mode-change" && e.data.mode) {
    const mode = e.data.mode === "simple" ? "simple" : "expert";
    document.documentElement.setAttribute("data-ui-mode", mode);
    publishHeaderContribution();
  }
});

applyEmbedded();

// Draggable sidebar/reader splitter (expert layout; hidden in simple mode).
initSidebarResize();

/**
 * @typedef {Object} ConceptFrontmatter
 * @property {string} [title]
 * @property {string} [description]
 * @property {unknown[]} [tags]
 */

/**
 * Shape of the `/api/concept` response. `okf_panel/app.py` returns a raw
 * dict with no pydantic response model, so any of these keys may simply be
 * absent rather than explicitly null.
 * @typedef {Object} ConceptDoc
 * @property {string} [id]
 * @property {ConceptFrontmatter} [frontmatter]
 * @property {string} [body]
 */

/**
 * @param {string} id
 * @returns {HTMLElement}
 */
function requireEl(id) {
  const node = document.getElementById(id);
  if (!node) throw new Error("okf_panel: missing required element #" + id);
  return node;
}

(function () {
  // -- DOM handles -----------------------------------------------------------
  const treeEl = requireEl("tree");
  const readerEl = requireEl("reader-content");
  const searchForm = requireEl("search-form");
  const searchInput = /** @type {HTMLInputElement} */ (requireEl("search-input"));
  const searchResultsEl = requireEl("search-results");
  const structureLink = document.getElementById("structure-link");

  initTree({ treeEl, structureLink, onSelect: loadConcept });
  initSearchResults({ containerEl: searchResultsEl, onSelect: selectConcept });

  // -------------------------------------------------------------------------
  // Render hook for task 3.2.
  //
  // Called once after every reading-pane render with the freshly-populated
  // container element. Task 3.2 will replace this no-op body to wire up
  // cross-link navigation. Do NOT inline post-render work elsewhere — keep it
  // funnelled through here so 3.2 has a single integration point.
  // -------------------------------------------------------------------------
  /**
   * @param {HTMLElement} container
   */
  function afterRender(container) {
    // Task 3.2: wire cross-link navigation for every anchor in the freshly
    // rendered container. Three classes of href:
    //   1. in-bundle cross-links  (/^\/.+\.md$/) → intercept, loadConcept + pushState
    //   2. external links         (http:// or https://) → target=_blank rel=noopener
    //   3. everything else        (#anchors, /dir/ directory links) → untouched
    //
    // loadConcept re-renders, which re-invokes afterRender, so links inside the
    // newly-rendered concept get wired recursively on each navigation.
    //
    // Double-wiring guard: each render replaces readerEl.innerHTML, so anchors
    // are always fresh nodes with no listeners. We still mark wired anchors with
    // data-okf-wired and skip already-marked ones, so this stays correct even if
    // afterRender is ever called twice on the same container.
    if (!container) return;
    const anchors = /** @type {NodeListOf<HTMLElement>} */ (
      container.querySelectorAll("a[href]")
    );
    anchors.forEach(function (a) {
      if (a.dataset.okfWired === "1") return;
      const href = a.getAttribute("href") || "";

      if (/^\/.+\.md$/.test(href)) {
        // In-bundle cross-link.
        a.dataset.okfWired = "1";
        a.addEventListener("click", function (ev) {
          ev.preventDefault();
          const id = href.replace(/^\//, "").replace(/\.md$/, "");
          loadConcept(id);
          history.pushState({ id: id }, "", "#" + id);
        });
      } else if (/^https?:\/\//.test(href)) {
        // External link — open in a new tab, no opener leakage. No preventDefault.
        a.dataset.okfWired = "1";
        a.setAttribute("target", "_blank");
        a.setAttribute("rel", "noopener");
      }
      // else: in-page #anchors or /dir/ directory links — leave default behavior.
    });
  }

  // Back/forward navigation between cross-linked concepts. Guard the initial
  // (null) state so popping to the entry point doesn't throw or re-load.
  window.addEventListener("popstate", function (ev) {
    if (ev.state && ev.state.id === STRUCTURE_MARKER) {
      // Pop back to the structure overview WITHOUT pushing a new entry.
      loadStructure({ push: false });
    } else if (ev.state && ev.state.id) {
      // Re-render the target concept WITHOUT pushing a new history entry.
      // (renderConcept re-applies the sidebar highlight + scroll-into-view.)
      loadConcept(ev.state.id);
    }
  });

  // -- sidebar / tree --------------------------------------------------------

  async function loadTree() {
    let data;
    try {
      const resp = await fetch("/api/concepts");
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      data = await resp.json();
    } catch {
      treeEl.innerHTML = "";
      treeEl.appendChild(
        el("p", { class: "muted", text: "Failed to load concepts." })
      );
      return;
    }
    renderTree(data.groups || []);
  }

  // -- reading pane ----------------------------------------------------------

  /**
   * @param {string} id
   */
  async function loadConcept(id) {
    renderMessage("Loading…");
    let resp;
    try {
      resp = await fetch("/api/concept?id=" + encodeURIComponent(id));
    } catch {
      renderMessage("Failed to load concept.");
      return;
    }

    if (resp.status === 404) {
      renderMessage('Concept not found: "' + id + '"');
      return;
    }
    if (!resp.ok) {
      renderMessage("Failed to load concept (HTTP " + resp.status + ").");
      return;
    }

    let doc;
    try {
      doc = await resp.json();
    } catch {
      renderMessage("Failed to parse concept.");
      return;
    }
    renderConcept(doc);
  }

  // Single render path for the reading pane — always ends by calling
  // afterRender(readerEl) so task 3.2 has one integration point.
  /**
   * @param {ConceptDoc} doc
   */
  function renderConcept(doc) {
    const fm = doc.frontmatter || {};
    const id = doc.id || "";
    const title = fm.title != null ? String(fm.title) : "";
    const description = fm.description != null ? String(fm.description) : "";
    const fallback = isFallback(id, title);

    // Keep the sidebar selection in sync with the page actually being shown,
    // regardless of how we got here (sidebar click, in-body cross-link,
    // structure-overview link, or back/forward). Single authoritative point.
    highlightActive(id);

    readerEl.innerHTML = "";

    // Heading: for fallback entries, show the full concept id instead of the
    // bare last-segment title.
    const heading = fallback ? id : title || id;
    readerEl.appendChild(el("h1", { class: "concept-title", text: heading }));

    // Description line — omitted for fallback entries (and when empty).
    if (!fallback && description) {
      readerEl.appendChild(
        el("p", { class: "concept-description", text: description })
      );
    }

    // Tags, if present in frontmatter.
    const tags = fm.tags;
    if (Array.isArray(tags) && tags.length > 0) {
      const tagWrap = el("div", { class: "concept-tags" });
      for (const tag of tags) {
        tagWrap.appendChild(el("span", { class: "tag", text: String(tag) }));
      }
      readerEl.appendChild(tagWrap);
    }

    // Markdown body, rendered via the vendored marked. The container MUST be
    // exactly class="osprey-md-rendered" so the gallery markdown CSS applies.
    //
    // TRUST MODEL: marked.parse output is assigned to innerHTML unsanitized.
    // This is safe under OKF's authoring model — bundles are admin-authored,
    // read-only facility knowledge served locally, the same trust assumption as
    // the artifact gallery / channel-finder markdown. If bundles ever accepted
    // untrusted authorship this would need a sanitizer (e.g. DOMPurify).
    const body = doc.body != null ? String(doc.body) : "";
    const bodyEl = el("div", { class: "osprey-md-rendered" });
    try {
      bodyEl.innerHTML = marked.parse(body);
    } catch {
      bodyEl.textContent = body;
    }
    readerEl.appendChild(bodyEl);

    afterRender(readerEl);
  }

  /**
   * @param {string} msg
   */
  function renderMessage(msg) {
    readerEl.innerHTML = "";
    readerEl.appendChild(el("p", { class: "muted", text: msg }));
    afterRender(readerEl);
  }

  // -- structure overview ----------------------------------------------------
  //
  // Fetches /api/structure (a markdown document) and renders it into the
  // reading pane as an .osprey-md-rendered container. Its concept links are in
  // the /<id>.md form, so afterRender() wires them as in-panel navigation.
  /**
   * @param {{push?: boolean}} [opts]
   */
  async function loadStructure(opts) {
    const push = !opts || opts.push !== false;
    highlightStructure();
    renderMessage("Loading…");

    let data;
    try {
      const resp = await fetch("/api/structure");
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      data = await resp.json();
    } catch {
      renderMessage("Failed to load knowledge base overview.");
      return;
    }

    readerEl.innerHTML = "";
    const container = el("div", { class: "osprey-md-rendered" });
    const md = data.markdown != null ? String(data.markdown) : "";
    try {
      container.innerHTML = marked.parse(md);
    } catch {
      container.textContent = md;
    }
    readerEl.appendChild(container);
    afterRender(container);

    if (push) {
      history.pushState({ id: STRUCTURE_MARKER }, "", "#" + STRUCTURE_MARKER);
    }
  }

  // -- search ----------------------------------------------------------------
  //
  // Rendering lives in search.js (see its header for the ranked/unranked
  // presentation decision); this only fetches and hands over the hits.

  /**
   * @param {string} query
   */
  async function runSearch(query) {
    const q = (query || "").trim();
    if (!q) {
      clearSearchResults();
      return;
    }

    let data;
    try {
      const resp = await fetch("/api/search?q=" + encodeURIComponent(q));
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      data = await resp.json();
    } catch {
      renderSearchMessage("Search failed.");
      return;
    }
    renderSearchResults(data.results || []);
  }

  // debounce is imported from the shared design-system dom.js (identical
  // trailing-edge behaviour to the local copy this replaces).
  const debouncedSearch = debounce(function () {
    runSearch(searchInput.value);
  }, 200);

  searchInput.addEventListener("input", debouncedSearch);
  searchForm.addEventListener("submit", function (ev) {
    ev.preventDefault();
    runSearch(searchInput.value);
  });

  // Embedded: the same box lives in the hub's tile bar and posts its text back
  // here. The hub already debounces before reporting, so this runs the search
  // directly rather than through debouncedSearch.
  onHeaderAction(function (id, value) {
    if (id === "search") runSearch(value || "");
  });

  if (structureLink) {
    structureLink.addEventListener("click", function (ev) {
      ev.preventDefault();
      loadStructure();
    });
  }

  // Simple-mode "Browse all pages" affordance — with the expert sidebar tree
  // hidden, this is the reading-focused mode's route back to the structure
  // overview (the plain doc list). Harmless in Expert (the button is hidden).
  const browseAllBtn = document.getElementById("browse-all");
  if (browseAllBtn) {
    browseAllBtn.addEventListener("click", function () {
      loadStructure();
    });
  }

  // -- bundle health (osprey addition) ---------------------------------------
  //
  // Surfaces the panel's own /api/bundle_health summary (broken cross-links /
  // frontmatter issues) as a small sidebar status line. Guarded/unconfigured
  // panels (503) simply hide the line.
  async function loadBundleHealth() {
    const footer = document.getElementById("bundle-health");
    if (!footer) return;

    let data;
    try {
      const resp = await fetch("/api/bundle_health");
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      data = await resp.json();
    } catch {
      footer.hidden = true;
      return;
    }

    footer.hidden = false;
    footer.innerHTML = "";
    if (data.ok) {
      footer.classList.remove("has-warnings");
      footer.appendChild(el("span", { class: "health-dot ok" }));
      footer.appendChild(el("span", { class: "health-text", text: "Bundle healthy" }));
    } else {
      footer.classList.add("has-warnings");
      const counts = data.counts || {};
      const parts = Object.keys(counts)
        .sort()
        .map(function (k) {
          return k + "=" + counts[k];
        });
      const total = data.total || 0;
      footer.appendChild(el("span", { class: "health-dot warn" }));
      footer.appendChild(
        el("span", {
          class: "health-text",
          text: total + " issue" + (total === 1 ? "" : "s") + ": " + parts.join(", "),
        })
      );
    }
  }

  // -- boot ------------------------------------------------------------------
  //
  // Load the sidebar + bundle-health line, then honour any in-panel deep-link
  // (URL hash → concept); otherwise show the structure overview as the default
  // reader content (instead of an empty pane).
  function bootFromParams() {
    const params = readPanelParams();
    if (params.concept) {
      // Seed a history entry with the target id so browser back/forward returns
      // to the deep-linked concept (popstate reads ev.state.id).
      history.replaceState({ id: params.concept }, "", "#" + params.concept);
      loadConcept(params.concept);
    } else {
      loadStructure({ push: false });
    }
  }

  loadTree();
  loadBundleHealth();
  bootFromParams();
  // Embedded: hand the search box to the host tile bar (Expert only).
  publishHeaderContribution();
})();
