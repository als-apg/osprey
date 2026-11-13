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

import { initTheme } from '/design-system/js/theme-manager.js';
import { applyEmbedded } from '/design-system/js/frame-params.js';
import { debounce } from '/design-system/js/dom.js';
import { contributeHeader, onHeaderAction } from '/design-system/js/header-contrib.js';
import { renderState } from '/design-system/js/panel-state.js';
import {
  el,
  FACILITY_SCHEME,
  facilityLinkView,
  GRAPH_MARKER,
  graphHash,
  graphTarget,
  hashChangeTarget,
  isFallback,
  readPanelParams,
  STRUCTURE_MARKER,
} from './helpers.js';
import { createGraphView } from './graph-view.js';
import { currentView, graphAvailable, initViews, showView } from './views.js';
import {
  initTree,
  renderTree,
  highlightActive,
  highlightStructure,
  selectConcept,
} from './tree.js';
import {
  initSearchResults,
  clearSearchResults,
  renderSearchMessage,
  renderSearchResults,
} from './search.js';
import {
  contentsPicked,
  initContents,
  resetContentsWidth,
  setContentsSearching,
  showContentsFor,
} from './contents.js';
import { initToolbar, showBundleHealth, showCrumb } from './toolbar.js';

// A follower of the Web Terminal hub: theme-boot.js already applied data-theme
// pre-paint, and this attaches the postMessage listener for the hub's live
// broadcasts.
initTheme({ role: 'follower' });

/**
 * The panel's menu, in the tile ⋯ under "This panel". Its one entry resets the
 * width of the open view's sidebar: the contents list, or the graph controls.
 */
const PANEL_MENU = {
  id: 'knowledge',
  items: [{ id: 'reset-width', label: 'Reset contents width' }],
};

/**
 * Resets the Graph view's sidebar width; set once the view is wired.
 * @type {(() => void) | null}
 */
let resetGraphWidth = null;

/**
 * Run an entry of the panel's menu: it comes back as the menu's id and the
 * entry's id.
 * @param {string} id
 * @param {string} [entry]
 */
function onPanelMenu(id, entry) {
  if (id !== PANEL_MENU.id || entry !== 'reset-width') return;
  if (currentView() === 'graph' && resetGraphWidth) resetGraphWidth();
  else resetContentsWidth();
}

applyEmbedded();

// The contents list (the sidebar): its toggle, divider and when it shows.
initContents();

onHeaderAction(onPanelMenu);

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
 * @property {import("./helpers.js").FacilityReferenceView[]} [facility_references]
 *   each facility link of the body with the records it names
 */

/**
 * The DOMPurify instance this page sanitises markdown HTML with, or `null`.
 *
 * `index.html` loads DOMPurify as a page global (CDN, or the copy vendored
 * into this panel's `static/vendor/` when offline); `null` when it did not
 * load.
 *
 * @param {Window} [win]
 * @returns {{ sanitize: (html: string) => string } | null}
 */
export function resolveSanitizer(win = window) {
  const own = /** @type {any} */ (win).DOMPurify;
  return own && typeof own.sanitize === 'function' ? own : null;
}

/**
 * The panel's one markdown-to-HTML sink: both the concept body and the
 * structure overview render through it.
 *
 * HTML is written only when BOTH the vendored `marked` and a DOMPurify
 * ({@link resolveSanitizer}) are available, and only DOMPurify's output ever
 * reaches `innerHTML` — bundle markdown is files on disk an agent can write,
 * so raw `marked` output is untrusted HTML (`<img onerror>`, `<script>`,
 * `javascript:` links). Missing either library, or either step throwing,
 * writes the source as inert `textContent` instead (the chat-render.js
 * precedent).
 *
 * @param {HTMLElement} target - its contents are replaced
 * @param {unknown} markdown - raw markdown; `null`/`undefined` render empty
 * @returns {void}
 */
export function renderMarkdownInto(target, markdown) {
  const text = markdown != null ? String(markdown) : '';
  const parser = /** @type {any} */ (globalThis).marked;
  const purify = resolveSanitizer();
  if (purify && parser && typeof parser.parse === 'function') {
    try {
      target.innerHTML = purify.sanitize(String(parser.parse(text)));
      return;
    } catch {
      // Fall through to inert text.
    }
  }
  target.textContent = text;
}

/**
 * @param {string} id
 * @returns {HTMLElement}
 */
function requireEl(id) {
  const node = document.getElementById(id);
  if (!node) throw new Error('okf_panel: missing required element #' + id);
  return node;
}

(function () {
  // -- DOM handles -----------------------------------------------------------
  const treeEl = requireEl('tree');
  const readerEl = requireEl('reader-content');
  const searchInput = /** @type {HTMLInputElement} */ (requireEl('search-input'));
  const searchResultsEl = requireEl('search-results');
  const structureLink = document.getElementById('structure-link');
  /** @type {Set<string> | null} */
  let knownConceptIds = null;
  /** Each concept's title, and each group's label by its id, once the tree loaded. */
  const conceptTitles = /** @type {Map<string, string>} */ (new Map());
  const groupLabels = /** @type {Map<string, string>} */ (new Map());
  /**
   * The open page: the concept's id (the tile's subtitle) and its title (the
   * breadcrumb's last crumb); null on the overview.
   * @type {{id: string, title: string} | null}
   */
  let openPage = null;
  /** @type {ConceptDoc["facility_references"] | null} */
  let currentFacilityReferences = null;

  initTree({ treeEl, structureLink, onSelect: pickConcept });
  initSearchResults({ containerEl: searchResultsEl, onSelect: selectConcept });
  initToolbar({ onBrowse: (group) => loadStructure({ section: group }) });

  // -- the two views ---------------------------------------------------------
  //
  // The Knowledge view (the contents list and the reader) and, when the panel
  // serves a facility graph, the Graph view. Each keeps its place while the
  // other is shown; the history holds one entry per switch.

  /**
   * What the Knowledge view shows: a concept's id or STRUCTURE_MARKER; null
   * until it has loaded something.
   * @type {string | null}
   */
  let knowledgeLoc = null;

  const graph = graphAvailable()
    ? createGraphView({
        onPage(id) {
          loadConcept(id);
          history.pushState({ id: id }, '', '#' + id);
        },
        // The Graph view's entry follows what it draws, so a reload and a
        // shared link reopen the same neighbourhood.
        onLocation(node) {
          if (currentView() !== 'graph') return;
          history.replaceState({ id: GRAPH_MARKER, node: node }, '', graphHash(node));
        },
      })
    : null;
  if (graph) resetGraphWidth = graph.resetWidth;

  /**
   * Show a view. The search field is the open view's, so a switch empties it.
   * @param {import('./views.js').PanelView} view
   */
  function enterView(view) {
    if (!showView(view)) return;
    searchInput.value = '';
    clearSearchResults();
    setContentsSearching(false);
    if (graph) graph.search('');
  }

  /**
   * Show the Graph view on a node's neighbourhood, or on its overview.
   * @param {string | null} node
   */
  function openGraph(node) {
    if (!graph) return;
    enterView('graph');
    graph.show(node);
  }

  /**
   * Show the Graph view on the node that stands for a facility record.
   * @param {{kind: string, id: string}} record
   */
  function showInGraph(record) {
    if (!graph) return;
    enterView('graph');
    history.pushState({ id: GRAPH_MARKER, node: null }, '', graphHash(null));
    graph.showRecord(record.kind, record.id);
  }

  initViews({
    onSwitch(view) {
      if (view === 'graph' && graph) {
        const node = graph.location();
        history.pushState({ id: GRAPH_MARKER, node: node }, '', graphHash(node));
        openGraph(node);
      } else if (knowledgeLoc === null) {
        loadStructure();
      } else {
        enterView('knowledge');
        history.pushState({ id: knowledgeLoc }, '', '#' + knowledgeLoc);
      }
    },
  });

  /**
   * A row of the contents list (the tree or a search result) was picked.
   * @param {string} id
   */
  function pickConcept(id) {
    loadConcept(id);
    contentsPicked();
  }

  /**
   * The label of the group a concept is in (its id's first segment), or null
   * for a top-level id and before the tree has loaded.
   * @param {string} id
   * @returns {string | null}
   */
  function groupOf(id) {
    const slash = id.indexOf('/');
    return slash > 0 ? (groupLabels.get(id.slice(0, slash)) ?? null) : null;
  }

  /**
   * Name the open page in the tile's subtitle (re-sent when the concept
   * changes) and in the breadcrumb.
   * @param {{id: string, title: string} | null} page null on the overview
   */
  function showOpenPage(page) {
    openPage = page;
    showCrumb(page && { group: groupOf(page.id), title: page.title });
  }

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
    // rendered container. Four classes of href:
    //   1. in-bundle cross-links  (/^\/.+\.md$/) → intercept, loadConcept + pushState
    //   2. external links         (http:// or https://) → target=_blank rel=noopener
    //   3. facility links         (facility:) → the linked record on hover; a click
    //                             shows the record in the Graph view when there is one
    //   4. everything else        (#anchors, /dir/ directory links) → untouched
    //
    // loadConcept re-renders, which re-invokes afterRender, so links inside the
    // newly-rendered concept get wired recursively on each navigation.
    //
    // Double-wiring guard: each render replaces readerEl.innerHTML, so anchors
    // are always fresh nodes with no listeners. We still mark wired anchors with
    // data-okf-wired and skip already-marked ones, so this stays correct even if
    // afterRender is ever called twice on the same container.
    if (!container) return;
    const anchors = /** @type {NodeListOf<HTMLElement>} */ (container.querySelectorAll('a[href]'));
    anchors.forEach(function (a) {
      if (a.dataset.okfWired === '1') return;
      const href = a.getAttribute('href') || '';

      if (/^\/.+\.md$/.test(href)) {
        // In-bundle cross-link.
        a.dataset.okfWired = '1';
        a.addEventListener('click', function (ev) {
          ev.preventDefault();
          const id = href.replace(/^\//, '').replace(/\.md$/, '');
          loadConcept(id);
          history.pushState({ id: id }, '', '#' + id);
        });
      } else if (/^https?:\/\//.test(href)) {
        // External link — open in a new tab, no opener leakage. No preventDefault.
        a.dataset.okfWired = '1';
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener');
      } else if (href.startsWith(FACILITY_SCHEME)) {
        const view = facilityLinkView(href, currentFacilityReferences);
        a.dataset.okfWired = '1';
        a.classList.add(view.dangling ? 'okf-facility-link--dangling' : 'okf-facility-link');
        const span = (currentFacilityReferences || []).find((entry) => entry.target === view.text);
        const record = graph && span && span.records ? span.records[0] : undefined;
        if (record) a.classList.add('okf-facility-link--graph');
        a.setAttribute('title', record ? view.title + '\nClick: show in the graph' : view.title);
        a.addEventListener('click', function (ev) {
          ev.preventDefault();
          if (record) showInGraph(record);
        });
      }
      // else: in-page #anchors or /dir/ directory links — leave default behavior.
    });
  }

  // Back/forward navigation between cross-linked concepts. Guard the initial
  // (null) state so popping to the entry point doesn't throw or re-load.
  window.addEventListener('popstate', function (ev) {
    if (ev.state && ev.state.id === GRAPH_MARKER) {
      openGraph(ev.state.node ?? null);
    } else if (ev.state && ev.state.id === STRUCTURE_MARKER) {
      // Pop back to the structure overview WITHOUT pushing a new entry.
      loadStructure({ push: false });
    } else if (ev.state && ev.state.id) {
      // Re-render the target concept WITHOUT pushing a new history entry.
      // (renderConcept re-applies the sidebar highlight + scroll-into-view.)
      loadConcept(ev.state.id);
    } else if (currentView() === 'graph') {
      // The entry the panel opened on holds no state: it is the knowledge
      // page that was showing before the Graph view.
      if (knowledgeLoc === null) loadStructure({ push: false });
      else enterView('knowledge');
    }
  });

  // The hub opens this panel on a concept by changing its fragment; an
  // in-page anchor changes it too, and names no concept.
  window.addEventListener('hashchange', function () {
    const params = readPanelParams();
    const inGraph = graph ? graphTarget(params.raw) : null;
    if (inGraph) {
      const state = history.state;
      const held = state && state.id === GRAPH_MARKER && (state.node ?? null) === inGraph.node;
      if (held) return;
      history.replaceState({ id: GRAPH_MARKER, node: inGraph.node }, '');
      openGraph(inGraph.node);
      return;
    }
    const target = hashChangeTarget(params, history.state, knownConceptIds);
    if (target === null) return;
    history.replaceState({ id: target }, '');
    if (target === STRUCTURE_MARKER) loadStructure({ push: false });
    else loadConcept(target);
  });

  // -- sidebar / tree --------------------------------------------------------

  async function loadTree() {
    let data;
    try {
      const resp = await fetch('/api/concepts');
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      data = await resp.json();
    } catch {
      treeEl.innerHTML = '';
      treeEl.appendChild(el('p', { class: 'muted', text: 'Failed to load concepts.' }));
      return;
    }
    /** @type {{id?: string, label?: string, concepts?: {id: string, title?: string}[]}[]} */
    const groups = data.groups || [];
    knownConceptIds = new Set();
    for (const group of groups) {
      if (group.id) groupLabels.set(group.id, group.label || group.id);
      for (const concept of group.concepts || []) {
        knownConceptIds.add(concept.id);
        conceptTitles.set(concept.id, concept.title || concept.id);
      }
    }
    renderTree(groups);
    // The breadcrumb names the open page's group once the groups are known.
    if (openPage) showOpenPage(openPage);
  }

  // -- reading pane ----------------------------------------------------------

  /**
   * One centred state in the reading pane, which then states no concept.
   * @param {Parameters<typeof renderState>[1]} view
   */
  function showReaderState(view) {
    delete readerEl.dataset.conceptId;
    renderState(readerEl, view);
  }

  /**
   * The reading pane could not get what it asked for: one centred state, with
   * the one next step.
   * @param {string} message
   * @param {() => void} retry
   */
  function renderLoadFailure(message, retry) {
    showReaderState({
      icon: 'triangle-alert',
      message,
      action: { label: 'Retry', onClick: retry },
    });
  }

  /**
   * @param {string} id
   */
  async function loadConcept(id) {
    enterView('knowledge');
    knowledgeLoc = id;
    showContentsFor('page');
    showOpenPage({ id, title: conceptTitles.get(id) ?? id });
    showReaderState({ icon: 'refresh', message: 'Loading the page' });
    const retry = () => loadConcept(id);
    let resp;
    try {
      resp = await fetch('/api/concept?id=' + encodeURIComponent(id));
    } catch {
      renderLoadFailure('The page did not load', retry);
      return;
    }

    if (resp.status === 404) {
      showReaderState({
        icon: 'search',
        message: 'No concept has this id',
        detail: id,
        action: { label: 'Browse', onClick: () => loadStructure() },
      });
      return;
    }

    let doc = null;
    if (resp.ok) {
      try {
        doc = await resp.json();
      } catch {
        /* unreadable: reported below */
      }
    }
    if (!doc) {
      renderLoadFailure('The page did not load', retry);
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
    const id = doc.id || '';
    const title = fm.title != null ? String(fm.title) : '';
    const description = fm.description != null ? String(fm.description) : '';
    const fallback = isFallback(id, title);

    // Keep the sidebar selection in sync with the page actually being shown,
    // regardless of how we got here (sidebar click, in-body cross-link,
    // structure-overview link, or back/forward). Single authoritative point.
    highlightActive(id);
    showOpenPage({ id, title: title || id });

    readerEl.innerHTML = '';

    // Heading: for fallback entries, show the full concept id instead of the
    // bare last-segment title.
    const heading = fallback ? id : title || id;
    readerEl.appendChild(el('h1', { class: 'concept-title', text: heading }));

    // Description line — omitted for fallback entries (and when empty).
    if (!fallback && description) {
      readerEl.appendChild(el('p', { class: 'concept-description', text: description }));
    }

    // Tags, if present in frontmatter.
    const tags = fm.tags;
    if (Array.isArray(tags) && tags.length > 0) {
      const tagWrap = el('div', { class: 'concept-tags' });
      for (const tag of tags) {
        tagWrap.appendChild(el('span', { class: 'tag', text: String(tag) }));
      }
      readerEl.appendChild(tagWrap);
    }

    // Markdown body. The container MUST be exactly class="osprey-md-rendered"
    // so the gallery markdown CSS applies. Rendered through the one sanitising
    // sink (renderMarkdownInto): a bundle is files on disk an agent can write.
    const bodyEl = el('div', { class: 'osprey-md-rendered' });
    renderMarkdownInto(bodyEl, doc.body);
    readerEl.appendChild(bodyEl);

    currentFacilityReferences = doc.facility_references || null;
    // The reader's statement of the concept it shows; present only while one is rendered.
    readerEl.dataset.conceptId = id;
    afterRender(readerEl);
  }

  // -- structure overview ----------------------------------------------------
  //
  // Fetches /api/structure (a markdown document) and renders it into the
  // reading pane as an .osprey-md-rendered container. Its concept links are in
  // the /<id>.md form, so afterRender() wires them as in-panel navigation.
  /**
   * @param {{push?: boolean, section?: string | null}} [opts]
   *   `section`: the label of a group, whose section the overview scrolls to
   */
  async function loadStructure(opts) {
    const push = !opts || opts.push !== false;
    enterView('knowledge');
    knowledgeLoc = STRUCTURE_MARKER;
    showContentsFor('browse');
    highlightStructure();
    showOpenPage(null);
    showReaderState({ icon: 'refresh', message: 'Loading the overview' });

    let data;
    try {
      const resp = await fetch('/api/structure');
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      data = await resp.json();
    } catch {
      renderLoadFailure('The overview did not load', () => loadStructure(opts));
      return;
    }

    delete readerEl.dataset.conceptId;
    readerEl.innerHTML = '';
    const container = el('div', { class: 'osprey-md-rendered' });
    renderMarkdownInto(container, data.markdown);
    readerEl.appendChild(container);
    afterRender(container);
    const section = opts && opts.section;
    if (section) {
      const heading = Array.from(container.querySelectorAll('h2')).find(
        (h) => (h.textContent || '').trim() === section
      );
      heading?.scrollIntoView({ block: 'start' });
    }

    if (push) {
      history.pushState({ id: STRUCTURE_MARKER }, '', '#' + STRUCTURE_MARKER);
    }
  }

  // -- search ----------------------------------------------------------------
  //
  // The toolbar's search field. Rendering lives in search.js (see its header
  // for the ranked/unranked presentation decision); this only fetches and
  // hands over the hits.

  /**
   * @param {string} query
   */
  async function runSearch(query) {
    const q = (query || '').trim();
    // In the Graph view the field searches the graph's nodes.
    if (currentView() === 'graph') {
      if (graph) graph.search(q);
      return;
    }
    // The results are drawn in the contents list, so a search shows it.
    setContentsSearching(q !== '');
    if (!q) {
      clearSearchResults();
      return;
    }

    let data;
    try {
      const resp = await fetch('/api/search?q=' + encodeURIComponent(q));
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      data = await resp.json();
    } catch {
      renderSearchMessage('Search failed.');
      return;
    }
    renderSearchResults(data.results || []);
  }

  // debounce is imported from the shared design-system dom.js (identical
  // trailing-edge behaviour to the local copy this replaces).
  const debouncedSearch = debounce(function () {
    runSearch(searchInput.value);
  }, 200);

  searchInput.addEventListener('input', debouncedSearch);
  // Enter searches at once, without waiting out the debounce.
  searchInput.addEventListener('keydown', function (ev) {
    if (ev.key === 'Enter') runSearch(searchInput.value);
  });

  if (structureLink) {
    structureLink.addEventListener('click', function (ev) {
      ev.preventDefault();
      loadStructure();
    });
  }

  // -- bundle health (osprey addition) ---------------------------------------
  //
  // The panel's own /api/bundle_health check of the served bundle (broken
  // cross-links, frontmatter issues), shown as the toolbar's status chip
  // (toolbar.js). A check that fails, an unconfigured panel's 503 included,
  // shows as failed, with the reason behind the chip.
  async function loadBundleHealth() {
    let resp;
    try {
      resp = await fetch('/api/bundle_health');
    } catch {
      showBundleHealth({ state: 'failed', reason: 'the panel did not answer' });
      return;
    }
    let data = null;
    try {
      data = await resp.json();
    } catch {
      /* unreadable: reported below */
    }
    if (resp.ok && data) {
      showBundleHealth({ state: 'checked', health: data });
      return;
    }
    const said = data && typeof data.error === 'string' ? data.error : 'no readable answer';
    showBundleHealth({ state: 'failed', reason: said + ' (HTTP ' + resp.status + ')' });
  }

  // -- boot ------------------------------------------------------------------
  //
  // Load the contents list and the bundle check, then honour any in-panel
  // deep-link (URL hash → concept); otherwise show the structure overview as
  // the default reader content (instead of an empty pane).
  function bootFromParams() {
    const params = readPanelParams();
    const inGraph = graph ? graphTarget(params.raw) : null;
    if (inGraph) {
      history.replaceState({ id: GRAPH_MARKER, node: inGraph.node }, '', graphHash(inGraph.node));
      openGraph(inGraph.node);
    } else if (params.concept) {
      // Seed a history entry with the target id so browser back/forward returns
      // to the deep-linked concept (popstate reads ev.state.id).
      history.replaceState({ id: params.concept }, '', '#' + params.concept);
      loadConcept(params.concept);
    } else {
      loadStructure({ push: false });
    }
  }

  loadTree();
  loadBundleHealth();
  bootFromParams();
  // Embedded: the tile bar's subtitle (on a page) and its ⋯ entry.
  contributeHeader([{ kind: 'menu', ...PANEL_MENU }]);
})();
