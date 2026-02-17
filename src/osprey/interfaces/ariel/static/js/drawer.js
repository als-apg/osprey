/**
 * Drawer infrastructure
 *
 * Generic open/close/toggle for slide-in panels.
 */

const backdrop = () => document.getElementById('drawer-backdrop');

/**
 * Open a drawer by ID.
 * @param {string} drawerId
 */
export function openDrawer(drawerId) {
  const drawer = document.getElementById(drawerId);
  if (!drawer) return;
  drawer.classList.add('open');
  const bd = backdrop();
  if (bd) bd.classList.add('visible');
}

/**
 * Close a drawer by ID (or all drawers if none specified).
 * @param {string} [drawerId]
 */
export function closeDrawer(drawerId) {
  if (drawerId) {
    const drawer = document.getElementById(drawerId);
    if (drawer) drawer.classList.remove('open');
  } else {
    document.querySelectorAll('.drawer.open').forEach(d => d.classList.remove('open'));
  }
  const bd = backdrop();
  if (bd) bd.classList.remove('visible');
}

/**
 * Toggle a drawer open/closed.
 * @param {string} drawerId
 */
export function toggleDrawer(drawerId) {
  const drawer = document.getElementById(drawerId);
  if (!drawer) return;
  if (drawer.classList.contains('open')) {
    closeDrawer(drawerId);
  } else {
    closeDrawer(); // close any other open drawer
    openDrawer(drawerId);
  }
}

/**
 * Initialize drawer system — wire up header buttons, close button, backdrop, Escape key.
 */
export function initDrawers() {
  // Header icon buttons with data-drawer attribute
  document.querySelectorAll('[data-drawer]').forEach(btn => {
    btn.addEventListener('click', () => {
      toggleDrawer(btn.dataset.drawer);
    });
  });

  // Close buttons inside drawers
  document.querySelectorAll('.drawer-close-btn').forEach(btn => {
    btn.addEventListener('click', () => closeDrawer());
  });

  // Backdrop click closes drawers
  const bd = backdrop();
  if (bd) {
    bd.addEventListener('click', () => closeDrawer());
  }

  // Escape key closes drawers (but not if a modal is open)
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const openDrawers = document.querySelectorAll('.drawer.open');
      if (openDrawers.length > 0) {
        e.preventDefault();
        closeDrawer();
      }
    }
  });
}
