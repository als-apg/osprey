// @ts-check
/**
 * OSPREY Artifact Gallery — sidebar rendering layer.
 *
 * Owns the filter bar, the shared gallery-card template, the sidebar
 * dispatcher (tree/activity mode renderers + their shared item handlers),
 * and the split-pane resize handle. Everything here reads/writes the
 * shared artifact list via state.js and formats via types.js.
 *
 * The split-pane resize handler has no dependency on anything outside its
 * own arguments, so it's a plain/pure export (`initSplitPaneResize`). The
 * rest needs two effects this module doesn't own — setting agent focus and
 * (re)rendering the preview pane / entering fullscreen, owned by preview.js's
 * preview renderer and wired through gallery.js — so
 * `createSidebarRenderer(callbacks)` injects them, mirroring
 * lattice_dashboard/render.js's createRenderer(callbacks) pattern.
 *
 * @module render
 */

import {
  getArtifacts,
  getSelectedArtifact,
  setSelectedArtifact,
  getActiveFilter,
  setActiveFilter,
  getFilteredArtifacts,
} from "./state.js";
import {
  getTypeRegistry,
  typeBadge,
  typeIcon,
  thumbnailHtml,
  escapeHtml,
  formatSize,
  formatTime,
  formatDate,
  isNewThisSession,
  requestColorPass,
} from "./types.js";

// ---- Split-Pane Resize ----

/**
 * Wire the sidebar/preview split-pane drag handle.
 * @param {HTMLElement|null} handle
 * @param {HTMLElement|null} sidebarEl
 * @returns {void}
 */
export function initSplitPaneResize(handle, sidebarEl) {
  if (!handle || !sidebarEl) return;
  /** @type {number} */
  let startX;
  /** @type {number} */
  let startWidth;
  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    startX = e.clientX;
    startWidth = sidebarEl.offsetWidth;
    /** @param {MouseEvent} ev */
    const onMove = (ev) => {
      const delta = ev.clientX - startX;
      const newW = Math.max(180, Math.min(startWidth + delta, window.innerWidth * 0.6));
      sidebarEl.style.width = newW + "px";
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });
}

// ---- Gallery Card HTML (shared by both sidebar modes in gallery layout) ----

/**
 * @param {any} a
 * @param {number} i
 * @returns {string}
 */
function galleryCardHtml(a, i) {
  const sel = getSelectedArtifact() && getSelectedArtifact().id === a.id ? " selected" : "";
  const pinnedCls = a.pinned ? " pinned" : "";
  return `
    <div class="gallery-card${sel}${pinnedCls}"
         data-id="${a.id}"
         data-type="${escapeHtml(a.category || a.artifact_type)}"
         style="animation-delay: ${i * 30}ms">
      <div class="gallery-card-thumb">${thumbnailHtml(a)}</div>
      <div class="gallery-card-info">
        <div class="gallery-card-title" title="${escapeHtml(a.title)}">
          ${a.pinned ? '<span class="pin-indicator" title="Pinned">&#128204;</span>' : ""}
          ${escapeHtml(a.title)}
        </div>
        <div class="gallery-card-meta">
          <span class="gallery-card-type">${typeBadge(a.category || a.artifact_type)}</span>
          <span class="gallery-card-time">${formatTime(a.timestamp)}</span>
          <span class="gallery-card-size">${formatSize(a.size_bytes)}</span>
        </div>
      </div>
    </div>`;
}

const chevronSvg = '<svg class="tree-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18l6-6-6-6"/></svg>';

// Session-start timestamp for the tree-mode "new" badge (isNewThisSession
// compares each artifact's timestamp against this). Computed once at this
// module's load time, same as gallery.js's own (now-removed) `_sessionStart`
// — both modules load within the same page load, so the sub-millisecond
// skew between the two is immaterial to the "is this new since I opened the
// gallery" feature this drives.
const _sessionStart = new Date().toISOString();

/**
 * @typedef {object} SidebarRenderCallbacks
 * @property {(artifact: any) => void} onSelect - fired right after a single-clicked item is marked selected (drives the still-gallery.js-owned setAsFocus POST /api/focus)
 * @property {() => void} onPreviewNeeded - fired once selection actually changes (not on a re-click of the already-selected item), to (re)render the preview pane
 * @property {(artifact: any) => void} onEnterFullscreen - fired on double-click, to enter fullscreen mode for that artifact
 */

/**
 * Create the gallery's sidebar renderer: filter bar, tree/activity mode
 * dispatch, and the drag-to-terminal/click/dblclick item handlers. Bound to
 * a small set of injected callbacks for the two effects (agent focus,
 * preview/fullscreen) still owned by gallery.js's not-yet-extracted Preview
 * Pane section.
 * @param {SidebarRenderCallbacks} callbacks
 */
export function createSidebarRenderer(callbacks) {
  /** @type {"tree"|"activity"} */
  let browseMode = "tree";
  /** @type {"list"|"gallery"} */
  let sidebarLayout = "list";
  // Guards wireFilterChips so the delegated #filter-bar click listener is
  // registered exactly once per renderer instance, no matter how many times
  // initFilterBar/rebuildTypeChips run over the page lifetime (refetch, SSE).
  let filterBarWired = false;

  /** @returns {"tree"|"activity"} */
  function getBrowseMode() { return browseMode; }
  /** @param {"tree"|"activity"} mode */
  function setBrowseMode(mode) { browseMode = mode; }
  /** @returns {"list"|"gallery"} */
  function getSidebarLayout() { return sidebarLayout; }
  /** @param {"list"|"gallery"} layout */
  function setSidebarLayout(layout) { sidebarLayout = layout; }

  // ---- Filter Bar ----

  function initFilterBar() {
    const filterBar = document.getElementById("filter-bar");
    if (!filterBar) return;
    rebuildTypeChips();
    wireFilterChips();
  }

  /** Rebuild all conditional filter chips based on current artifacts. */
  function rebuildTypeChips() {
    const filterBar = document.getElementById("filter-bar");
    if (!filterBar) return;

    // --- Pinned chip: show only when count > 0 ---
    const pinnedCount = getArtifacts().filter((a) => a.pinned).length;

    const pinnedChip = /** @type {HTMLElement|null} */ (filterBar.querySelector('[data-filter="pinned"]'));

    if (pinnedChip) {
      /** @type {any} */ (pinnedChip).hidden = pinnedCount === 0;
      const countEl = pinnedChip.querySelector(".chip-count");
      if (countEl) countEl.textContent = String(pinnedCount || "");
    }

    // --- Type chips: show only types that have artifacts ---
    const typesContainer = document.getElementById("filter-type-chips");
    if (!typesContainer) return;

    const presentTypes = new Set(getArtifacts().map((a) => a.category || a.artifact_type));

    // If current filter no longer has artifacts, reset to "all"
    if (getActiveFilter() === "pinned" && pinnedCount === 0) {
      setActiveFilter("all");
    } else if (getActiveFilter() !== "all"
        && getActiveFilter() !== "pinned" && !presentTypes.has(getActiveFilter())) {
      setActiveFilter("all");
    }

    typesContainer.innerHTML = "";
    // Prefer category-based chips when registry provides categories
    const chipSource = getTypeRegistry().categories || getTypeRegistry().artifact_types;
    if (chipSource) {
      Object.entries(chipSource).forEach(([type, info]) => {
        if (!presentTypes.has(type)) return;
        const count = getArtifacts().filter((a) => (a.category || a.artifact_type) === type).length;
        const chip = document.createElement("button");
        chip.className = "filter-chip type-chip";
        chip.dataset.filter = type;
        // typeIcon(type) is intentionally NOT escaped here: it returns
        // hardcoded SVG markup from an internal map keyed by `icons[type] ||
        // icons.text` — the `type` argument never reaches the output, so
        // there is nothing agent-controlled in its return value (audited
        // 2026-07-07). The label text below IS agent-controlled (registry
        // label or the raw category/artifact_type) and must be escaped.
        chip.innerHTML = `<span class="chip-icon">${typeIcon(type)}</span>${escapeHtml((/** @type {any} */ (info)).label || type)} <span class="chip-count">${count}</span>`;
        typesContainer.appendChild(chip);
      });
    }

    updateFilterBarActive();
  }

  /** Wire a single delegated click listener on #filter-bar (registered once). */
  function wireFilterChips() {
    if (filterBarWired) return;
    const filterBar = document.getElementById("filter-bar");
    if (!filterBar) return;
    filterBar.addEventListener("click", (e) => {
      const chip = /** @type {HTMLElement|null} */ (/** @type {HTMLElement} */ (e.target).closest(".filter-chip"));
      if (!chip) return;
      setActiveFilter(chip.dataset.filter || "all");
      updateFilterBarActive();
      renderSidebar();
    });
    filterBarWired = true;
  }

  function updateFilterBarActive() {
    const filterBar = document.getElementById("filter-bar");
    if (!filterBar) return;
    filterBar.querySelectorAll(".filter-chip").forEach((chip) => {
      chip.classList.toggle("active", getActiveFilter() === /** @type {HTMLElement} */ (chip).dataset.filter);
    });
  }

  // ---- Sidebar Rendering (dispatcher + tree/activity renderers) ----

  function renderSidebar() {
    const sidebarBody = document.getElementById("sidebar-body");
    if (!sidebarBody) return;
    const searchInput = /** @type {HTMLInputElement|null} */ (document.getElementById("search"));

    const filtered = getFilteredArtifacts(searchInput ? searchInput.value.trim().toLowerCase() : "");

    if (filtered.length === 0) {
      sidebarBody.innerHTML = `
        <div class="sidebar-empty">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <rect x="3" y="3" width="7" height="7" rx="1"/>
            <rect x="14" y="3" width="7" height="7" rx="1"/>
            <rect x="3" y="14" width="7" height="7" rx="1"/>
            <rect x="14" y="14" width="7" height="7" rx="1"/>
          </svg>
          <span>${(searchInput && searchInput.value) || getActiveFilter() !== "all" ? "No matches" : "No artifacts yet"}</span>
        </div>
      `;
      return;
    }

    if (browseMode === "tree") {
      renderTreeMode(filtered);
    } else {
      renderActivityMode(filtered);
    }
    requestColorPass();
  }

  // ---- Tree Mode (group by type) ----

  /** @param {any[]} items */
  function renderTreeMode(items) {
    const sidebarBody = document.getElementById("sidebar-body");
    if (!sidebarBody) return;

    /** @type {Record<string, any[]>} */
    const groups = {};
    items.forEach((a) => {
      const groupKey = a.category || a.artifact_type;
      if (!groups[groupKey]) groups[groupKey] = [];
      groups[groupKey].push(a);
    });

    const sortedTypes = Object.keys(groups).sort((a, b) => {
      const diff = groups[b].length - groups[a].length;
      return diff !== 0 ? diff : a.localeCompare(b);
    });

    const isGallery = sidebarLayout === "gallery";
    let html = "";
    let globalIdx = 0;

    sortedTypes.forEach((type) => {
      const typeArtifacts = groups[type];

      if (isGallery) {
        html += `
          <div class="tree-section" data-type="${escapeHtml(type)}">
            <div class="gallery-section-header" data-type="${escapeHtml(type)}">
              ${chevronSvg}
              <span class="tree-section-icon">${typeIcon(type)}</span>
              <span>${typeBadge(type)}</span>
              <span class="tree-section-count">${typeArtifacts.length}</span>
            </div>
            <div class="tree-section-items sidebar-gallery">
              ${typeArtifacts.map((a) => galleryCardHtml(a, globalIdx++)).join("")}
            </div>
          </div>`;
      } else {
        html += `
          <div class="tree-section" data-type="${escapeHtml(type)}">
            <div class="tree-section-header" data-type="${escapeHtml(type)}">
              ${chevronSvg}
              <span class="tree-section-icon">${typeIcon(type)}</span>
              <span>${typeBadge(type)}</span>
              <span class="tree-section-count">${typeArtifacts.length}</span>
            </div>
            <div class="tree-section-items">
              ${typeArtifacts
                .map(
                  (a, i) => `
                <div class="tree-item${getSelectedArtifact() && getSelectedArtifact().id === a.id ? " selected" : ""}${a.pinned ? " pinned" : ""}"
                     data-id="${a.id}"
                     style="animation-delay: ${i * 30}ms">
                  ${a.pinned ? '<span class="pin-indicator" title="Pinned">&#128204;</span>' : ""}
                  <span class="tree-item-icon">${typeIcon(a.artifact_type)}</span>
                  <span class="tree-item-name" title="${escapeHtml(a.title)}">${escapeHtml(a.title)}</span>
                  ${isNewThisSession(a, _sessionStart) ? '<span class="tree-item-badge new">new</span>' : ""}
                  <span class="tree-item-size">${formatSize(a.size_bytes)}</span>
                </div>`
                )
                .join("")}
            </div>
          </div>`;
      }
    });

    sidebarBody.innerHTML = html;
    attachSidebarHandlers();
  }

  // ---- Activity Mode (chronological timeline) ----

  /** @param {any[]} items */
  function renderActivityMode(items) {
    const sidebarBody = document.getElementById("sidebar-body");
    if (!sidebarBody) return;

    /** @type {Record<string, any[]>} */
    const dateGroups = {};
    items.forEach((a) => {
      const label = formatDate(a.timestamp);
      if (!dateGroups[label]) dateGroups[label] = [];
      dateGroups[label].push(a);
    });

    const isGallery = sidebarLayout === "gallery";
    let html = "";
    let itemIndex = 0;

    Object.entries(dateGroups).forEach(([label, group]) => {
      html += `<div class="timeline-group">`;
      html += `<div class="timeline-group-label">${label}</div>`;

      if (isGallery) {
        html += `<div class="sidebar-gallery">`;
        group.forEach((a) => { html += galleryCardHtml(a, itemIndex++); });
        html += `</div>`;
      } else {
        group.forEach((a) => {
          html += `
            <div class="timeline-item${getSelectedArtifact() && getSelectedArtifact().id === a.id ? " selected" : ""}${a.pinned ? " pinned" : ""}"
                 data-id="${a.id}"
                 data-type="${escapeHtml(a.category || a.artifact_type)}"
                 style="animation-delay: ${itemIndex * 25}ms">
              <span class="timeline-dot"></span>
              <div class="timeline-item-body">
                <div class="timeline-item-title" title="${escapeHtml(a.title)}">
                  ${a.pinned ? '<span class="pin-indicator">&#128204;</span>' : ""}
                  ${escapeHtml(a.title)}
                </div>
                <div class="timeline-item-meta">
                  <span class="timeline-item-type">${typeBadge(a.category || a.artifact_type)}</span>
                  <span class="timeline-item-time">${formatTime(a.timestamp)}</span>
                </div>
              </div>
            </div>`;
          itemIndex++;
        });
      }

      html += `</div>`;
    });

    sidebarBody.innerHTML = html;
    attachSidebarHandlers();
  }

  // ---- Shared item handlers (unified: click/dblclick/drag-to-terminal) ----

  function attachSidebarHandlers() {
    const sidebarBody = document.getElementById("sidebar-body");
    if (!sidebarBody) return;

    // Tree/gallery section toggle
    sidebarBody.querySelectorAll(".tree-section-header, .gallery-section-header").forEach((header) => {
      header.addEventListener("click", () => {
        /** @type {Element} */ (header.parentElement).classList.toggle("collapsed");
      });
    });

    // Item click, double-click (fullscreen), drag-and-drop (send to terminal)
    const clickables = ".tree-item, .timeline-item, .gallery-card";
    sidebarBody.querySelectorAll(clickables).forEach((el) => {
      el.addEventListener("click", (e) => {
        if (/** @type {HTMLElement} */ (e.target).closest(".tree-section-header, .gallery-section-header")) return;
        const id = /** @type {HTMLElement} */ (el).dataset.id;
        const a = getArtifacts().find((x) => x.id === id);
        if (a) {
          const alreadySelected = getSelectedArtifact()?.id === a.id;
          setSelectedArtifact(a);
          callbacks.onSelect(a);
          sidebarBody.querySelectorAll(clickables).forEach((item) => item.classList.remove("selected"));
          el.classList.add("selected");
          if (!alreadySelected) callbacks.onPreviewNeeded();
        }
      });

      // Fullscreen: double-click
      el.addEventListener("dblclick", (e) => {
        e.preventDefault();
        const id = /** @type {HTMLElement} */ (el).dataset.id;
        const a = getArtifacts().find((x) => x.id === id);
        if (!a) return;
        setSelectedArtifact(a);
        callbacks.onEnterFullscreen(a);
      });

      // Drag-and-drop: drag artifact to terminal to paste reference
      /** @type {HTMLElement} */ (el).draggable = true;
      el.addEventListener("dragstart", (e) => {
        const id = /** @type {HTMLElement} */ (el).dataset.id;
        const a = getArtifacts().find((x) => x.id === id);
        if (!a) return;
        const text = `Please have a look at _agent_data/artifacts/${a.filename}`;
        const dragEvent = /** @type {DragEvent} */ (e);
        /** @type {DataTransfer} */ (dragEvent.dataTransfer).setData("text/plain", text);
        /** @type {DataTransfer} */ (dragEvent.dataTransfer).effectAllowed = "copy";
      });
    });
  }

  return {
    initFilterBar,
    rebuildTypeChips,
    renderSidebar,
    getBrowseMode,
    setBrowseMode,
    getSidebarLayout,
    setSidebarLayout,
  };
}
