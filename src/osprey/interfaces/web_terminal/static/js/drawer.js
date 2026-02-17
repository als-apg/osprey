/* OSPREY Web Terminal — Drawer Infrastructure */

let activeDrawer = null;

/**
 * Open a drawer panel by ID.
 * Only one drawer can be open at a time — opening a new one closes the current.
 */
export function openDrawer(drawerId) {
  if (activeDrawer === drawerId) return;
  if (activeDrawer) closeDrawer();

  const drawer = document.getElementById(drawerId);
  const backdrop = document.getElementById('drawer-backdrop');
  if (!drawer) return;

  activeDrawer = drawerId;
  drawer.classList.add('open');
  if (backdrop) backdrop.classList.add('open');

  // Restore persisted width
  const saved = localStorage.getItem('osprey-drawer-width');
  if (saved) {
    const width = parseInt(saved, 10);
    if (width >= 320) {
      drawer.style.width = width + 'px';
      drawer.style.maxWidth = 'none';
    }
  }

  // Mark header button as active
  const btn = document.querySelector(`[data-drawer="${drawerId}"]`);
  if (btn) btn.classList.add('active');

  // Fire open event for panel-specific init
  drawer.dispatchEvent(new CustomEvent('drawer:open'));

  // Also fire tab-activate on the currently active tab panel
  const activePanel = drawer.querySelector('.drawer-tab-panel.active');
  if (activePanel) {
    activePanel.dispatchEvent(new CustomEvent('drawer:tab-activate'));
  }
}

/**
 * Close the currently open drawer.
 * Returns false if close was cancelled by an unsaved-changes guard.
 */
export function closeDrawer() {
  if (!activeDrawer) return true;

  const drawer = document.getElementById(activeDrawer);

  // Check unsaved changes guard
  if (drawer && !checkUnsavedChanges(drawer)) return false;

  const backdrop = document.getElementById('drawer-backdrop');

  if (drawer) {
    drawer.classList.remove('open');
    drawer.dispatchEvent(new CustomEvent('drawer:close'));
  }
  if (backdrop) backdrop.classList.remove('open');

  // Clear header button active state
  const btn = document.querySelector(`[data-drawer="${activeDrawer}"]`);
  if (btn) btn.classList.remove('active');

  activeDrawer = null;
  return true;
}

/**
 * Toggle a drawer open/closed.
 */
export function toggleDrawer(drawerId) {
  if (activeDrawer === drawerId) {
    closeDrawer();
  } else {
    openDrawer(drawerId);
  }
}

/**
 * Check if any tab panel has unsaved changes.
 * Returns true if safe to proceed, false if user cancelled.
 */
function checkUnsavedChanges(drawer) {
  // Look for a registered guard
  if (drawer._unsavedGuard && typeof drawer._unsavedGuard === 'function') {
    return drawer._unsavedGuard();
  }
  return true;
}

/**
 * Register an unsaved-changes guard function on a drawer.
 * The guard should return true if safe to proceed, false to cancel.
 */
export function registerUnsavedGuard(drawerId, guardFn) {
  const drawer = document.getElementById(drawerId);
  if (drawer) drawer._unsavedGuard = guardFn;
}

/**
 * Initialize drawer infrastructure: backdrop click, Escape key, close buttons,
 * tab switching, and resize handles.
 */
export function initDrawers() {
  // Backdrop click to close
  const backdrop = document.getElementById('drawer-backdrop');
  if (backdrop) {
    backdrop.addEventListener('click', closeDrawer);
  }

  // Escape key to close
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && activeDrawer) {
      closeDrawer();
    }
  });

  // Close buttons
  document.querySelectorAll('.drawer-close-btn').forEach((btn) => {
    btn.addEventListener('click', closeDrawer);
  });

  // Header icon buttons
  document.querySelectorAll('.header-icon-btn[data-drawer]').forEach((btn) => {
    btn.addEventListener('click', () => {
      toggleDrawer(btn.dataset.drawer);
    });
  });

  // Tab switching
  initDrawerTabs();

  // Resize handles
  initDrawerResize();
}

/**
 * Initialize tab switching for all drawers with tabs.
 */
function initDrawerTabs() {
  document.querySelectorAll('.drawer-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      const drawer = tab.closest('.drawer');
      if (!drawer) return;

      const targetId = tab.dataset.tab;
      const targetPanel = document.getElementById(targetId);
      if (!targetPanel) return;

      // Already active?
      if (tab.classList.contains('active')) return;

      // Check unsaved changes on the currently active panel
      if (!checkUnsavedChanges(drawer)) return;

      // Deactivate all tabs and panels in this drawer
      drawer.querySelectorAll('.drawer-tab').forEach((t) => t.classList.remove('active'));
      drawer.querySelectorAll('.drawer-tab-panel').forEach((p) => p.classList.remove('active'));

      // Activate the target
      tab.classList.add('active');
      targetPanel.classList.add('active');

      // Fire tab-activate event
      targetPanel.dispatchEvent(new CustomEvent('drawer:tab-activate'));
    });
  });
}

/**
 * Initialize resize handles for all drawers.
 */
function initDrawerResize() {
  document.querySelectorAll('.drawer-resize-handle').forEach((handle) => {
    const drawer = handle.closest('.drawer');
    if (!drawer) return;

    let isDragging = false;
    let startX = 0;
    let startWidth = 0;

    handle.addEventListener('mousedown', (e) => {
      isDragging = true;
      startX = e.clientX;
      startWidth = drawer.getBoundingClientRect().width;

      document.body.classList.add('drawer-resizing');
      handle.classList.add('dragging');
      e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
      if (!isDragging) return;

      // Drawer is on the right, so dragging left increases width
      const dx = startX - e.clientX;
      const maxWidth = window.innerWidth * 0.9;
      const newWidth = Math.max(320, Math.min(maxWidth, startWidth + dx));

      drawer.style.width = newWidth + 'px';
      drawer.style.maxWidth = 'none';
    });

    document.addEventListener('mouseup', () => {
      if (!isDragging) return;
      isDragging = false;

      document.body.classList.remove('drawer-resizing');
      handle.classList.remove('dragging');

      // Persist width
      const width = drawer.getBoundingClientRect().width;
      localStorage.setItem('osprey-drawer-width', String(Math.round(width)));
    });
  });
}

/**
 * Get the currently active drawer ID, or null.
 */
export function getActiveDrawer() {
  return activeDrawer;
}
