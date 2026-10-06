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
  getPickedIds,
  setPicked,
  togglePicked,
  clearPicked,
  getPickAnchor,
  setPickAnchor,
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

// ---- Picking several rows ----

/**
 * The inclusive run of ids between `anchorId` and `targetId`, in list order,
 * whichever of the two comes first. Just the target when the anchor is not in
 * the list.
 * @param {string[]} orderedIds
 * @param {string} anchorId
 * @param {string} targetId
 * @returns {string[]}
 */
export function pickRange(orderedIds, anchorId, targetId) {
  const from = orderedIds.indexOf(anchorId);
  const to = orderedIds.indexOf(targetId);
  if (from < 0 || to < 0) return [targetId];
  return orderedIds.slice(Math.min(from, to), Math.max(from, to) + 1);
}

/**
 * The class and ARIA suffix a row template adds for a picked artifact.
 * @param {any} a
 * @param {Set<string>} picked
 * @returns {{cls: string, aria: string}}
 */
function pickMarkup(a, picked) {
  return picked.has(a.id) ? { cls: " picked", aria: ' aria-selected="true"' } : { cls: "", aria: "" };
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
  const pick = pickMarkup(a, getPickedIds());
  return `
    <div class="gallery-card${sel}${pinnedCls}${pick.cls}"${pick.aria}
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
// module's load time, same as simple-view.js's own `_sessionStart`
// — both modules load within the same page load, so the sub-millisecond
// skew between the two is immaterial to the "is this new since I opened the
// gallery" feature this drives.
const _sessionStart = new Date().toISOString();

/**
 * @typedef {object} SidebarRenderCallbacks
 * @property {(artifact: any) => void} onSelect - fired right after a single-clicked item is marked selected (drives the still-gallery.js-owned setAsFocus POST /api/focus)
 * @property {() => void} onPreviewNeeded - fired once selection actually changes (not on a re-click of the already-selected item), to (re)render the preview pane
 * @property {(artifact: any) => void} onEnterFullscreen - fired on double-click, to enter fullscreen mode for that artifact
 * @property {() => void} [onPicksChanged] - fired after every render and every Shift/Cmd/Ctrl/plain click, so the pick bar follows the picks
 */

/**
 * Whether a scroll container is within `threshold` pixels of its end. 200 px
 * is about two list rows, so the next page is requested before the operator
 * reaches the end rather than after.
 * @param {{scrollHeight: number, scrollTop: number, clientHeight: number}} el
 * @param {number} [threshold]
 * @returns {boolean}
 */
export function isNearListEnd(el, threshold = 200) {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

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
      keepShownPicks(filtered);
      callbacks.onPicksChanged?.();
      return;
    }

    keepShownPicks(filtered);
    if (browseMode === "tree") {
      renderTreeMode(filtered);
    } else {
      renderActivityMode(filtered);
    }
    requestColorPass();
    callbacks.onPicksChanged?.();
  }

  /**
   * Drop picks for rows this render will not show, so a bulk delete never
   * reaches an artifact the operator cannot see.
   * @param {any[]} shown
   */
  function keepShownPicks(shown) {
    const picked = getPickedIds();
    if (picked.size === 0) return;
    const shownIds = new Set(shown.map((a) => a.id));
    const kept = [...picked].filter((id) => shownIds.has(id));
    if (kept.length !== picked.size) setPicked(kept);
  }

  // ---- Tree Mode (group by type, pinned promoted to the top) ----

  /**
   * @param {any} a
   * @param {number} i
   * @returns {string}
   */
  function treeItemHtml(a, i) {
    const pick = pickMarkup(a, getPickedIds());
    return `
                <div class="tree-item${getSelectedArtifact() && getSelectedArtifact().id === a.id ? " selected" : ""}${a.pinned ? " pinned" : ""}${pick.cls}"${pick.aria}
                     data-id="${a.id}"
                     style="animation-delay: ${i * 30}ms">
                  ${a.pinned ? '<span class="pin-indicator" title="Pinned">&#128204;</span>' : ""}
                  <span class="tree-item-icon">${typeIcon(a.artifact_type)}</span>
                  <span class="tree-item-name" title="${escapeHtml(a.title)}">${escapeHtml(a.title)}</span>
                  ${isNewThisSession(a, _sessionStart) ? '<span class="tree-item-badge new">new</span>' : ""}
                  ${a.origin === "demo" ? '<span class="tree-item-badge demo">example</span>' : ""}
                  <span class="tree-item-size">${formatSize(a.size_bytes)}</span>
                </div>`;
  }

  const exampleSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>';
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
    // Shipped examples (`origin: "demo"`) sit in their own section at the
    // bottom, outside pinning and the type groups: real work first.
    const exampleItems = items.filter((a) => a.origin === "demo");
    const ownItems = items.filter((a) => a.origin !== "demo");
    const pinnedItems = ownItems.filter((a) => a.pinned);
    const unpinned = ownItems.filter((a) => !a.pinned);

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

    if (exampleItems.length > 0) {
      html += treeSectionHtml({
        type: "examples", icon: exampleSvg, label: "Examples",
        items: exampleItems, isGallery, itemHtml,
      });
    }

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
      // A shipped example carries the day the gallery first started, which
      // is nobody's activity: it gets its own trailing group instead.
      const label = a.origin === "demo" ? "Examples" : formatDate(a.timestamp);
      if (!dateGroups[label]) dateGroups[label] = [];
      dateGroups[label].push(a);
    });
    if (dateGroups.Examples) {
      const examples = dateGroups.Examples;
      delete dateGroups.Examples;
      dateGroups.Examples = examples;
    }

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
          const pick = pickMarkup(a, getPickedIds());
          html += `
            <div class="timeline-item${getSelectedArtifact() && getSelectedArtifact().id === a.id ? " selected" : ""}${a.pinned ? " pinned" : ""}${pick.cls}"${pick.aria}
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
                  ${a.origin === "demo" ? '<span class="tree-item-badge demo">example</span>' : ""}
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
    const body = sidebarBody;

    /** Reflect the picks on the rendered rows in place and tell the pick bar. */
    function updatePickClasses() {
      const picked = getPickedIds();
      /** @type {NodeListOf<HTMLElement>} */ (body.querySelectorAll(clickables)).forEach((row) => {
        const on = picked.has(row.dataset.id || "");
        row.classList.toggle("picked", on);
        if (on) row.setAttribute("aria-selected", "true");
        else row.removeAttribute("aria-selected");
      });
      callbacks.onPicksChanged?.();
    }

    /** @returns {string[]} the ids of the rows shown, in sidebar order, skipping collapsed sections */
    function shownOrder() {
      return Array.from(/** @type {NodeListOf<HTMLElement>} */ (body.querySelectorAll(clickables)))
        .filter((row) => !row.closest(".tree-section.collapsed"))
        .map((row) => row.dataset.id || "");
    }

    sidebarBody.querySelectorAll(clickables).forEach((el) => {
      el.addEventListener("click", (e) => {
        if (/** @type {HTMLElement} */ (e.target).closest(".tree-section-header, .gallery-section-header")) return;
        const id = /** @type {HTMLElement} */ (el).dataset.id;
        if (!id) return;
        const mouse = /** @type {MouseEvent} */ (e);

        // Shift-click: pick the run from the anchor to this row; the anchor stays.
        if (mouse.shiftKey) {
          window.getSelection()?.removeAllRanges();
          const anchor = getPickAnchor() ?? getSelectedArtifact()?.id ?? id;
          setPicked(pickRange(shownOrder(), anchor, id));
          updatePickClasses();
          return;
        }

        // Cmd/Ctrl-click: add or remove this one row; it becomes the anchor.
        if (mouse.metaKey || mouse.ctrlKey) {
          const selected = getSelectedArtifact();
          if (getPickedIds().size === 0 && selected) setPicked([selected.id]);
          togglePicked(id);
          setPickAnchor(id);
          updatePickClasses();
          return;
        }

        clearPicked();
        setPickAnchor(id);
        updatePickClasses();
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
