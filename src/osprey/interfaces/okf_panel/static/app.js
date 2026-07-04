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
 * /static/app.js becomes fetch("/panel/okf/api/concept") in the browser.
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
import { applyEmbedded } from "/design-system/js/frame-params.js";
import { debounce } from "/design-system/js/dom.js";

// Panel embedded in the Web Terminal hub: apply the hub's broadcast theme and
// follow live `osprey-theme-change` messages. theme-boot.js already applied
// data-theme pre-paint; this attaches the follower's postMessage listener
// (replacing the legacy `theme:set` the panel's earlier TODO expected).
initTheme({ role: "follower" });

applyEmbedded();

(function () {
  // -- DOM handles -----------------------------------------------------------
  const treeEl = document.getElementById("tree");
  const readerEl = document.getElementById("reader-content");
  const searchForm = document.getElementById("search-form");
  const searchInput = document.getElementById("search-input");
  const searchResultsEl = document.getElementById("search-results");
  const structureLink = document.getElementById("structure-link");

  // History marker for the structure overview (B.3): a hash distinct from any
  // concept id so popstate can tell the overview apart from a concept view.
  const STRUCTURE_MARKER = "__structure";

  // -------------------------------------------------------------------------
  // Render hook for task 3.2.
  //
  // Called once after every reading-pane render with the freshly-populated
  // container element. Task 3.2 will replace this no-op body to wire up
  // cross-link navigation. Do NOT inline post-render work elsewhere — keep it
  // funnelled through here so 3.2 has a single integration point.
  // -------------------------------------------------------------------------
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
    const anchors = container.querySelectorAll("a[href]");
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

  // -- helpers ---------------------------------------------------------------

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        if (k === "class") node.className = attrs[k];
        else if (k === "text") node.textContent = attrs[k];
        else node.setAttribute(k, attrs[k]);
      }
    }
    if (children) {
      for (const child of children) {
        if (child) node.appendChild(child);
      }
    }
    return node;
  }

  // A fallback (filesystem-derived) concept is one whose title is just the
  // last path segment of its id — those have no real human title/description.
  function isFallback(id, title) {
    if (!id || !title) return false;
    const last = String(id).split("/").pop();
    return title === last;
  }

  // -- sidebar / tree --------------------------------------------------------

  async function loadTree() {
    let data;
    try {
      const resp = await fetch("/api/concepts");
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      data = await resp.json();
    } catch (err) {
      treeEl.innerHTML = "";
      treeEl.appendChild(
        el("p", { class: "muted", text: "Failed to load concepts." })
      );
      return;
    }
    renderTree(data.groups || []);
  }

  function renderTree(groups) {
    treeEl.innerHTML = "";
    for (const group of groups) {
      const section = el("section", { class: "group" });
      section.appendChild(
        el("h2", { class: "group-label", text: group.label || group.id || "" })
      );

      const concepts = group.concepts || [];
      if (concepts.length === 0) {
        section.appendChild(
          el("p", { class: "muted empty-group", text: "(no concepts)" })
        );
      } else {
        const list = el("ul", { class: "concept-list" });
        for (const concept of concepts) {
          const link = el("a", {
            class: "concept-link",
            href: "#",
            text: concept.title || concept.id,
            title: concept.description || "",
          });
          link.dataset.conceptId = concept.id;
          link.addEventListener("click", function (ev) {
            ev.preventDefault();
            selectConcept(concept.id);
          });
          const li = el("li", null, [link]);
          li.dataset.conceptId = concept.id;
          list.appendChild(li);
        }
        section.appendChild(list);
      }
      treeEl.appendChild(section);
    }
  }

  function highlightActive(id) {
    if (structureLink) structureLink.classList.remove("active");
    const links = treeEl.querySelectorAll(".concept-link");
    let activeLink = null;
    links.forEach(function (link) {
      if (link.dataset.conceptId === id) {
        link.classList.add("active");
        activeLink = link;
      } else {
        link.classList.remove("active");
      }
    });
    // Scroll the active entry into view so the user can always see where they
    // are in the tree — important when jumping via an in-body cross-link to a
    // concept far down the (scrolled) sidebar.
    if (activeLink) activeLink.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  // Mark the structure-overview entry active and clear any concept highlight.
  function highlightStructure() {
    const links = treeEl.querySelectorAll(".concept-link");
    links.forEach(function (link) {
      link.classList.remove("active");
    });
    if (structureLink) structureLink.classList.add("active");
  }

  function selectConcept(id) {
    // The sidebar highlight is applied authoritatively in renderConcept (which
    // every navigation path funnels through), so just trigger the load here.
    loadConcept(id);
  }

  // -- reading pane ----------------------------------------------------------

  async function loadConcept(id) {
    renderMessage("Loading…");
    let resp;
    try {
      resp = await fetch("/api/concept?id=" + encodeURIComponent(id));
    } catch (err) {
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
    } catch (err) {
      renderMessage("Failed to parse concept.");
      return;
    }
    renderConcept(doc);
  }

  // Single render path for the reading pane — always ends by calling
  // afterRender(readerEl) so task 3.2 has one integration point.
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
    } catch (err) {
      bodyEl.textContent = body;
    }
    readerEl.appendChild(bodyEl);

    afterRender(readerEl);
  }

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
  async function loadStructure(opts) {
    const push = !opts || opts.push !== false;
    highlightStructure();
    renderMessage("Loading…");

    let data;
    try {
      const resp = await fetch("/api/structure");
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      data = await resp.json();
    } catch (err) {
      renderMessage("Failed to load knowledge base overview.");
      return;
    }

    readerEl.innerHTML = "";
    const container = el("div", { class: "osprey-md-rendered" });
    const md = data.markdown != null ? String(data.markdown) : "";
    try {
      container.innerHTML = marked.parse(md);
    } catch (err) {
      container.textContent = md;
    }
    readerEl.appendChild(container);
    afterRender(container);

    if (push) {
      history.pushState({ id: STRUCTURE_MARKER }, "", "#" + STRUCTURE_MARKER);
    }
  }

  // -- search ----------------------------------------------------------------

  function clearSearchResults() {
    searchResultsEl.hidden = true;
    searchResultsEl.innerHTML = "";
  }

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
    } catch (err) {
      searchResultsEl.hidden = false;
      searchResultsEl.innerHTML = "";
      searchResultsEl.appendChild(
        el("p", { class: "muted", text: "Search failed." })
      );
      return;
    }
    renderSearchResults(data.results || []);
  }

  function renderSearchResults(results) {
    searchResultsEl.hidden = false;
    searchResultsEl.innerHTML = "";

    if (results.length === 0) {
      searchResultsEl.appendChild(
        el("p", { class: "muted", text: "No matches." })
      );
      return;
    }

    const list = el("ul", { class: "result-list" });
    for (const r of results) {
      const item = el("li", { class: "result-item" });
      const link = el("a", {
        class: "result-link",
        href: "#",
        text: r.title || r.id,
      });
      link.dataset.conceptId = r.id;
      link.addEventListener("click", function (ev) {
        ev.preventDefault();
        selectConcept(r.id);
      });
      item.appendChild(link);
      if (r.snippet) {
        item.appendChild(
          el("p", { class: "result-snippet", text: String(r.snippet) })
        );
      }
      list.appendChild(item);
    }
    searchResultsEl.appendChild(list);
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

  if (structureLink) {
    structureLink.addEventListener("click", function (ev) {
      ev.preventDefault();
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
    } catch (err) {
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

  // -- panel parameters / deep-link (osprey addition) ------------------------
  //
  // Generalizable panel-parameter shape. Today the only param is the deep-link
  // target carried in the panel's OWN URL hash (e.g. "#control-system/channel-
  // finding"). Keeping this as a small structured reader — rather than one-off
  // hash parsing scattered through boot — is deliberate: when the framework-wide
  // iframe URL-parameter convention lands (uniform theme / deep-link target /
  // etc. for every embedded panel), this one function is where it plugs in, with
  // no change to the navigation code below. Cross-boundary "the web terminal
  // opens the KNOWLEDGE tab on a concept" stays OUT of scope until then.
  function readPanelParams() {
    const hash = (location.hash || "").replace(/^#/, "");
    const params = { concept: null, structure: false, raw: hash };
    if (!hash) return params;
    if (hash === STRUCTURE_MARKER) {
      params.structure = true;
      return params;
    }
    try {
      params.concept = decodeURIComponent(hash);
    } catch (err) {
      params.concept = hash;
    }
    return params;
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
})();
