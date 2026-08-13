// @ts-check
/**
 * OSPREY Artifact Gallery — sidebar rendering layer.
 *
 * Owns the shared gallery-card template and the sidebar dispatcher
 * (tree/activity mode renderers + their shared item handlers). Tree mode
 * promotes pinned artifacts into a "Pinned" section at the top of the
 * tree — a promotion, not a filter, so the rest of the collection stays
 * in view. Everything here reads/writes the shared artifact list via
 * state.js and formats via types.js. (The browse split's orientation and
 * divider live in browse-layout.js.)
 *
 * Rendering needs two effects this module doesn't own — setting agent focus
 * and (re)rendering the preview pane / entering fullscreen, owned by
 * preview.js's preview renderer and wired through gallery.js — so
 * `createSidebarRenderer(callbacks)` injects them, mirroring
 * lattice_dashboard/render.js's createRenderer(callbacks) pattern.
 *
 * @module render
 */

import {
  getArtifacts,
  getSelectedArtifact,
  setSelectedArtifact,
  getFilteredArtifacts,
} from "./state.js";
import {
  typeBadge,
  typeIcon,
  thumbnailHtml,
  escapeHtml,
  formatSize,
  formatTime,
  formatDate,
  isNewThisSession,
  requestColorPass,
  artifactPath,
} from "./types.js";

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
 * Create the gallery's sidebar renderer: tree/activity mode dispatch and
 * the drag-to-terminal/click/dblclick item handlers. Bound to a small set
 * of injected callbacks for the two effects (agent focus,
 * preview/fullscreen) still owned by gallery.js's not-yet-extracted Preview
 * Pane section.
 * @param {SidebarRenderCallbacks} callbacks
 */
export function createSidebarRenderer(callbacks) {
  /** @type {"tree"|"activity"} */
  let browseMode = "tree";
  /** @type {"list"|"gallery"} */
  let sidebarLayout = "list";

  /** @returns {"tree"|"activity"} */
  function getBrowseMode() { return browseMode; }
  /** @param {"tree"|"activity"} mode */
  function setBrowseMode(mode) { browseMode = mode; }
  /** @returns {"list"|"gallery"} */
  function getSidebarLayout() { return sidebarLayout; }
  /** @param {"list"|"gallery"} layout */
  function setSidebarLayout(layout) { sidebarLayout = layout; }

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
          <span>${searchInput && searchInput.value ? "No matches" : "No artifacts yet"}</span>
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

  // ---- Tree Mode (group by type, pinned promoted to the top) ----

  /**
   * @param {any} a
   * @param {number} i
   * @returns {string}
   */
  function treeItemHtml(a, i) {
    return `
                <div class="tree-item${getSelectedArtifact() && getSelectedArtifact().id === a.id ? " selected" : ""}${a.pinned ? " pinned" : ""}"
                     data-id="${a.id}"
                     style="animation-delay: ${i * 30}ms">
                  ${a.pinned ? '<span class="pin-indicator" title="Pinned">&#128204;</span>' : ""}
                  <span class="tree-item-icon">${typeIcon(a.artifact_type)}</span>
                  <span class="tree-item-name" title="${escapeHtml(a.title)}">${escapeHtml(a.title)}</span>
                  ${isNewThisSession(a, _sessionStart) ? '<span class="tree-item-badge new">new</span>' : ""}
                  <span class="tree-item-size">${formatSize(a.size_bytes)}</span>
                </div>`;
  }

  const pinSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 17v5"/><path d="M9 10.76V7a2 2 0 00-1-1.73l-.5-.27A2 2 0 016.5 3.27V3h11v.27a2 2 0 01-1 1.73l-.5.27A2 2 0 0015 7v3.76a2 2 0 001 1.74l.5.27a2 2 0 011 1.73V15H6.5v-.5a2 2 0 011-1.73l.5-.27a2 2 0 001-1.74z"/></svg>';

  /**
   * A collapsible tree/gallery section. `label`/`icon` arrive as prebuilt
   * markup (typeBadge output or the fixed pin icon), never raw agent data.
   * @param {object} spec
   * @param {string} spec.type       section key for data-type (escaped here)
   * @param {string} spec.icon       icon markup
   * @param {string} spec.label      label markup
   * @param {any[]} spec.items       artifacts in the section
   * @param {boolean} spec.isGallery gallery layout?
   * @param {(a: any) => string} spec.itemHtml
   * @returns {string}
   */
  function treeSectionHtml({ type, icon, label, items, isGallery, itemHtml }) {
    const headerCls = isGallery ? "gallery-section-header" : "tree-section-header";
    const itemsCls = isGallery ? "tree-section-items sidebar-gallery" : "tree-section-items";
    return `
          <div class="tree-section" data-type="${escapeHtml(type)}">
            <div class="${headerCls}" data-type="${escapeHtml(type)}">
              ${chevronSvg}
              <span class="tree-section-icon">${icon}</span>
              <span>${label}</span>
              <span class="tree-section-count">${items.length}</span>
            </div>
            <div class="${itemsCls}">
              ${items.map((a) => itemHtml(a)).join("")}
            </div>
          </div>`;
  }

  /** @param {any[]} items */
  function renderTreeMode(items) {
    const sidebarBody = document.getElementById("sidebar-body");
    if (!sidebarBody) return;

    // Pinned artifacts are PROMOTED into their own top section (they do not
    // repeat inside their type groups) — the type groups hold the rest.
    const pinnedItems = items.filter((a) => a.pinned);
    const unpinned = items.filter((a) => !a.pinned);

    /** @type {Record<string, any[]>} */
    const groups = {};
    unpinned.forEach((a) => {
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
    /** @param {any} a @returns {string} */
    const itemHtml = (a) => (isGallery ? galleryCardHtml(a, globalIdx++) : treeItemHtml(a, globalIdx++));

    if (pinnedItems.length > 0) {
      html += treeSectionHtml({
        type: "pinned", icon: pinSvg, label: "Pinned",
        items: pinnedItems, isGallery, itemHtml,
      });
    }

    sortedTypes.forEach((type) => {
      html += treeSectionHtml({
        type, icon: typeIcon(type), label: typeBadge(type),
        items: groups[type], isGallery, itemHtml,
      });
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
        const text = `Please have a look at ${artifactPath(a)}`;
        const dragEvent = /** @type {DragEvent} */ (e);
        /** @type {DataTransfer} */ (dragEvent.dataTransfer).setData("text/plain", text);
        /** @type {DataTransfer} */ (dragEvent.dataTransfer).effectAllowed = "copy";
      });
    });
  }

  return {
    renderSidebar,
    getBrowseMode,
    setBrowseMode,
    getSidebarLayout,
    setSidebarLayout,
  };
}
