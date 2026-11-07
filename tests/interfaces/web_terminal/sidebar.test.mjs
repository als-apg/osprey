// @ts-check
/**
 * The sidebar shell (sidebar.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar.test.mjs
 *
 * What the shell adds to the rows index.html ships and panel-rail.js renders:
 * the glyphs (empty `svg[data-icon]` placeholders filled in place), the
 * platform's key hints, the labels toggle (collapsible.js, under the scoped
 * `osprey-sidebar` key sidebar-boot.js reads), the Agent row's session-tree
 * disclosure (kept under the scoped `osprey-sidebar-open` key), the collapsed
 * sidebar's sessions flyout, the Agent row's waiting badge, and the fades of
 * a group that scrolls.
 *
 * The fixture is the sidebar block and the flyout from index.html, copied with
 * the one Jinja expression (the brand's href) written out. Nothing is mocked:
 * icons.js, collapsible.js, popover.js and the chat bus (chat-events.js) are
 * the real modules. Module state (the open lists, the waiting count, the
 * popover registry) is isolated by vi.resetModules() + dynamic import per test.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { byId, qs } from '../_support/dom.mjs';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/sidebar.js')} */
let sidebar;
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js')} */
let chatEvents;
/** @type {typeof import('../../../src/osprey/interfaces/design_system/static/js/icons.js')} */
let icons;

const SCOPE_ATTR = 'data-osprey-storage-scope';

/** index.html's `nav#sidebar` and the body-level sessions flyout. */
const SIDEBAR_HTML = `
    <nav class="sidebar" id="sidebar" aria-label="Panels">
      <div class="sidebar-head">
        <a class="sidebar-brand" href="/" aria-label="OSPREY">
          <svg class="sidebar-brand-mark" viewBox="0 0 16 8" shape-rendering="crispEdges" aria-hidden="true" focusable="false"><path fill="currentColor" d="M0 0h2v1h-2zM14 0h2v1h-2zM1 1h3v1h-3zM12 1h3v1h-3zM2 2h4v1h-4zM10 2h4v1h-4zM3 3h4v1h-4zM9 3h4v1h-4zM4 4h8v1h-8zM6 5h4v1h-4zM7 6h2v1h-2zM6 7h1v1h-1zM9 7h1v1h-1z"/></svg>
          <span class="sidebar-brand-word">OSPREY</span>
        </a>
        <button class="osprey-icon-btn" id="sidebar-toggle" type="button"
                aria-label="Hide sidebar labels" aria-expanded="true" aria-controls="sidebar"
                data-tip-detail="Switch between the full sidebar and icons only">
          <svg class="osprey-ic" data-icon="panel-left" aria-hidden="true"></svg>
        </button>
      </div>

      <div class="sidebar-target" id="sidebar-target"></div>

      <button class="osprey-search" id="command-palette-btn" type="button"
              aria-label="Open command palette" data-tip="Search"
              data-tip-detail="Search settings, panels and actions">
        <svg class="osprey-ic" data-icon="search" aria-hidden="true"></svg>
        <span class="sidebar-search-label">Search</span>
        <kbd class="osprey-kbd" data-chord="K" aria-hidden="true">&#8984;K</kbd>
      </button>

      <div class="sidebar-sections" id="panel-rail" role="list">
        <div class="sidebar-group" data-group="workspace" role="listitem"
             aria-labelledby="sidebar-workspace-label">
          <div class="osprey-section-label sidebar-group-head" hidden>
            <span class="sidebar-group-name" id="sidebar-workspace-label">Sessions</span>
          </div>

          <div class="sidebar-row" id="sidebar-agent-wrap">
            <button class="osprey-nav-row parent panel-rail-button disabled" id="sidebar-agent-row"
                    data-panel-id="terminal" type="button" aria-label="Sessions" title="Sessions" data-tip="Sessions"
                    aria-expanded="true" aria-controls="sidebar-session-tree">
              <svg class="osprey-ic osprey-nav-icon" data-icon="agent" aria-hidden="true"></svg>
              <span class="osprey-nav-label">Sessions</span>
              <span class="sidebar-open-hint" aria-hidden="true">Open</span>
              <span class="osprey-nav-badge" data-agent-badge hidden></span>
            </button>
            <div class="sidebar-kids osprey-tree" id="sidebar-session-tree" data-session-tree
                 role="tree" aria-label="Sessions"></div>
          </div>
        </div>

        <div class="sidebar-group" data-group="tools" role="listitem" aria-labelledby="sidebar-tools-label">
          <div class="osprey-section-label sidebar-group-head">
            <span class="sidebar-group-name" id="sidebar-tools-label">Apps</span>
          </div>
          <button class="osprey-nav-row add" id="panel-add-btn" type="button" aria-label="Add panel"
                  aria-haspopup="dialog" aria-expanded="false" aria-controls="panel-add-menu" data-tip="Add panel"
                  data-tip-detail="Show a hidden panel or add one by URL">
            <svg class="osprey-ic osprey-nav-icon" data-icon="plus" aria-hidden="true"></svg>
            <span class="osprey-nav-label">Add panel</span>
          </button>
        </div>
      </div>

      <div class="sidebar-foot">
        <a class="osprey-nav-row" id="panel-docs-link" target="_blank" rel="noopener"
           title="Documentation" aria-label="Docs" data-tip="Docs" hidden>
          <svg class="osprey-ic osprey-nav-icon" data-icon="help" aria-hidden="true"></svg>
          <span class="osprey-nav-label">Docs</span>
        </a>
        <button class="osprey-nav-row" id="panel-feedback-btn" type="button"
                title="Send feedback" aria-label="Feedback" data-tip="Feedback">
          <svg class="osprey-ic osprey-nav-icon" data-icon="message" aria-hidden="true"></svg>
          <span class="osprey-nav-label">Feedback</span>
        </button>
        <button class="osprey-nav-row" id="more-btn" type="button" aria-label="More"
                aria-haspopup="menu" aria-expanded="false" aria-controls="more-menu" data-tip="More">
          <svg class="osprey-ic osprey-nav-icon" data-icon="menu" aria-hidden="true"></svg>
          <span class="osprey-nav-label">More</span>
        </button>
      </div>
    </nav>

  <div class="osprey-popover sidebar-flyout" id="sidebar-sessions-flyout" role="dialog" aria-label="Sessions" hidden>
    <div class="sidebar-flyout-head"><b>Sessions</b></div>
    <div class="sidebar-kids osprey-tree" data-session-tree role="tree" aria-label="Sessions"></div>
  </div>`;

/** @param {string} platform */
function setPlatform(platform) {
  Object.defineProperty(navigator, 'platform', { value: platform, configurable: true });
}

/** Render the fixture and import a fresh copy of the modules under test. */
async function load() {
  document.body.innerHTML = SIDEBAR_HTML;
  vi.resetModules();
  sidebar = await import(`${JS}/sidebar.js`);
  chatEvents = await import(`${JS}/chat-events.js`);
  icons = await import('/design-system/js/icons.js');
}

/** Load the fixture and wire it, as app.js does at boot. */
async function boot() {
  await load();
  const wired = sidebar.initSidebar();
  if (!wired) throw new Error('initSidebar found no #sidebar');
  return wired;
}

const agentRow = () => byId('sidebar-agent-row');
const tree = () => byId('sidebar-session-tree');
const flyout = () => byId('sidebar-sessions-flyout');
const agentBadge = () => qs(agentRow(), '[data-agent-badge]');

/** @param {HTMLElement} el @param {string} key @returns {KeyboardEvent} */
function press(el, key) {
  const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true });
  el.dispatchEvent(event);
  return event;
}

/** The stored open lists under `key`, parsed. @param {string} key */
const storedOpen = (key) => JSON.parse(window.localStorage.getItem(key) ?? 'null');

/** Collapse to icons only, through the labels toggle. */
const collapse = () => byId('sidebar-toggle').click();

beforeEach(() => {
  window.localStorage.clear();
  const root = document.documentElement;
  for (const attr of ['data-sidebar', 'data-agent', SCOPE_ATTR]) root.removeAttribute(attr);
  setPlatform('Linux x86_64');
});

afterEach(() => {
  setPlatform('');
  document.body.innerHTML = '';
});

/* ---- icons and key hints ------------------------------------------------ */

describe('hydrateIcons', () => {
  test('fills an empty placeholder in place, keeping the element', async () => {
    await load();
    document.body.innerHTML =
      '<svg class="osprey-ic osprey-nav-icon" data-icon="search" aria-hidden="true"></svg>';
    const placeholder = qs(document, 'svg[data-icon]');

    sidebar.hydrateIcons();

    expect(qs(document, 'svg[data-icon]')).toBe(placeholder);
    const glyph = icons.icon('search');
    expect(placeholder.childElementCount).toBe(glyph.childElementCount);
    expect(placeholder.innerHTML).toBe(glyph.innerHTML);
    expect(placeholder.getAttribute('viewBox')).toBe(glyph.getAttribute('viewBox'));
    expect(placeholder.getAttribute('aria-hidden')).toBe('true');
    expect(placeholder.getAttribute('focusable')).toBe('false');
    // The page's own classes stay: `.osprey-ic` sized it from the first paint.
    expect(placeholder.classList.contains('osprey-nav-icon')).toBe(true);
  });

  test('leaves a placeholder that already has a glyph alone', async () => {
    await load();
    document.body.innerHTML = '<svg data-icon="search"><circle r="1"></circle></svg>';
    const svg = qs(document, 'svg');

    sidebar.hydrateIcons();

    expect(svg.childElementCount).toBe(1);
    expect(svg.firstElementChild?.tagName.toLowerCase()).toBe('circle');
  });

  test('hydrates only under the root it is given', async () => {
    await load();
    document.body.innerHTML =
      '<div id="in"><svg data-icon="plus"></svg></div><div id="out"><svg data-icon="plus"></svg></div>';

    sidebar.hydrateIcons(byId('in'));

    expect(qs(byId('in'), 'svg').childElementCount).toBeGreaterThan(0);
    expect(qs(byId('out'), 'svg').childElementCount).toBe(0);
  });
});

describe('paintChords', () => {
  test('writes ⌘ on macOS', async () => {
    await load();
    setPlatform('MacIntel');
    sidebar.paintChords();
    expect(qs(document, '#command-palette-btn kbd').textContent).toBe('⌘K');
  });

  test('writes Ctrl+ elsewhere', async () => {
    await load();
    setPlatform('Linux x86_64');
    sidebar.paintChords();
    expect(qs(document, '#command-palette-btn kbd').textContent).toBe('Ctrl+K');
  });
});

/* ---- boot ---------------------------------------------------------------- */

describe('initSidebar', () => {
  test('is a no-op on a page without #sidebar', async () => {
    await load();
    document.body.innerHTML = '<div id="elsewhere"></div>';
    expect(sidebar.initSidebar()).toBeNull();
  });

  test('fills every glyph the page ships and paints the key hints', async () => {
    await boot();
    for (const svg of document.querySelectorAll('svg[data-icon]')) {
      expect(svg.childElementCount).toBeGreaterThan(0);
    }
    expect(qs(document, '#command-palette-btn kbd').textContent).toBe('Ctrl+K');
  });
});

/* ---- width --------------------------------------------------------------- */

describe('labels toggle', () => {
  test('flips html[data-sidebar] and stores it under the scoped key', async () => {
    document.documentElement.setAttribute(SCOPE_ATTR, 'alice');
    const wired = await boot();

    collapse();
    expect(document.documentElement.getAttribute('data-sidebar')).toBe('collapsed');
    expect(wired.isCollapsed()).toBe(true);
    expect(window.localStorage.getItem(`${sidebar.SIDEBAR_STORAGE_KEY}--alice`)).toBe('collapsed');
    expect(window.localStorage.getItem(sidebar.SIDEBAR_STORAGE_KEY)).toBeNull();

    collapse();
    expect(document.documentElement.getAttribute('data-sidebar')).toBe('expanded');
    expect(wired.isCollapsed()).toBe(false);
    expect(window.localStorage.getItem(`${sidebar.SIDEBAR_STORAGE_KEY}--alice`)).toBe('expanded');
  });

  test('stores under the bare key without a scope', async () => {
    await boot();
    collapse();
    expect(window.localStorage.getItem('osprey-sidebar')).toBe('collapsed');
  });
});

/* ---- disclosure ---------------------------------------------------------- */

describe('disclosure', () => {
  test('a first visit opens the session tree', async () => {
    await load();
    tree().hidden = true;
    agentRow().setAttribute('aria-expanded', 'false');

    sidebar.initSidebar();

    expect(tree().hidden).toBe(false);
    expect(agentRow().getAttribute('aria-expanded')).toBe('true');
    expect(agentRow().getAttribute('aria-controls')).toBe('sidebar-session-tree');
  });

  test('the shipped sidebar has no Artifacts or Script runs row, filter or menu', async () => {
    await boot();
    expect(document.getElementById('sidebar-artifacts-row')).toBeNull();
    expect(document.getElementById('sidebar-runs-row')).toBeNull();
    expect(
      document.querySelector('[data-art-filter], [data-art-menu], #sidebar-artifacts-menu')
    ).toBeNull();
  });

  test('the open tree persists per user under the scoped key', async () => {
    document.documentElement.setAttribute(SCOPE_ATTR, 'alice');
    await boot();
    const key = `${sidebar.OPEN_LISTS_STORAGE_KEY}--alice`;

    press(agentRow(), 'ArrowLeft');
    expect(tree().hidden).toBe(true);
    expect(storedOpen(key)).toEqual([]);
    expect(window.localStorage.getItem(sidebar.OPEN_LISTS_STORAGE_KEY)).toBeNull();

    // The next page load paints what was stored.
    await boot();
    expect(tree().hidden).toBe(true);
    expect(agentRow().getAttribute('aria-expanded')).toBe('false');

    press(agentRow(), 'ArrowRight');
    expect(storedOpen(key)).toEqual(['agent']);
  });

  test('another user on the same browser keeps the defaults', async () => {
    window.localStorage.setItem('osprey-sidebar-open--alice', JSON.stringify([]));
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
    await boot();
    expect(tree().hidden).toBe(false);
  });

  test('an unreadable stored value falls back to the defaults', async () => {
    for (const raw of ['not json', '{"agent": false}']) {
      window.localStorage.setItem('osprey-sidebar-open', raw);
      await boot();
      expect(tree().hidden).toBe(false);
    }
  });

  test('a stored list keeps only the rows it knows (an old ["artifacts"] closes the tree)', async () => {
    window.localStorage.setItem('osprey-sidebar-open', '["artifacts", "runs", 3]');
    await boot();
    expect(tree().hidden).toBe(true);
  });

  test("the expanded Agent row's click is left to its activation, which folds through setListOpen", async () => {
    await boot();
    const activate = vi.fn();
    // panel-menu-policy.js's activation decides what the click does.
    agentRow().addEventListener('click', activate);

    agentRow().click();
    expect(activate).toHaveBeenCalledOnce();
    expect(tree().hidden).toBe(false);
    expect(sidebar.isListOpen('agent')).toBe(true);

    sidebar.setListOpen('agent', false);
    expect(tree().hidden).toBe(true);
    expect(agentRow().getAttribute('aria-expanded')).toBe('false');
    expect(sidebar.isListOpen('agent')).toBe(false);
    expect(storedOpen('osprey-sidebar-open')).toEqual([]);

    sidebar.setListOpen('agent', true);
    expect(tree().hidden).toBe(false);
    expect(storedOpen('osprey-sidebar-open')).toEqual(['agent']);
  });

  test('the Agent row carries no chevron of its own', async () => {
    await boot();
    expect(agentRow().querySelector('[data-disc], .osprey-nav-disc')).toBeNull();
    expect(agentRow().getAttribute('aria-expanded')).toBe('true');
    expect(agentRow().getAttribute('aria-controls')).toBe('sidebar-session-tree');
  });

  test('ArrowRight opens and ArrowLeft closes the tree from the keyboard', async () => {
    await boot();
    const left = press(agentRow(), 'ArrowLeft');
    expect(left.defaultPrevented).toBe(true);
    expect(tree().hidden).toBe(true);
    press(agentRow(), 'ArrowRight');
    expect(tree().hidden).toBe(false);
    expect(press(agentRow(), 'ArrowDown').defaultPrevented).toBe(false);
  });
});

/* ---- collapsed ----------------------------------------------------------- */

describe('collapsed', () => {
  test('the arrow keys leave the tree alone', async () => {
    await boot();
    collapse();
    const left = press(agentRow(), 'ArrowLeft');
    expect(left.defaultPrevented).toBe(false);
    expect(tree().hidden).toBe(false);
  });

  test("the Agent row opens the sessions flyout beside the sidebar, not the row's pane half", async () => {
    await boot();
    const activate = vi.fn();
    agentRow().addEventListener('click', activate);
    collapse();

    agentRow().click();
    expect(flyout().hidden).toBe(false);
    expect(agentRow().getAttribute('aria-controls')).toBe('sidebar-sessions-flyout');
    expect(agentRow().getAttribute('aria-expanded')).toBe('true');
    expect(activate).not.toHaveBeenCalled();
    // The flyout is the tree, not a change to it.
    expect(storedOpen('osprey-sidebar-open')).toBeNull();

    agentRow().click();
    expect(flyout().hidden).toBe(true);
    expect(agentRow().getAttribute('aria-expanded')).toBe('false');
  });

  test('back to labels closes the flyout and points the row at its tree again', async () => {
    await boot();
    collapse();
    agentRow().click();
    expect(flyout().hidden).toBe(false);

    collapse();

    expect(document.documentElement.getAttribute('data-sidebar')).toBe('expanded');
    expect(flyout().hidden).toBe(true);
    expect(agentRow().getAttribute('aria-controls')).toBe('sidebar-session-tree');
    expect(agentRow().getAttribute('aria-expanded')).toBe('true');
    expect(tree().hidden).toBe(false);
  });
});

/* ---- the Sessions row ---------------------------------------------------- */

describe('the Sessions row', () => {
  /** @param {'open'|'closed'} state */
  function agentPane(state) {
    document.documentElement.setAttribute('data-agent', state);
    document.dispatchEvent(new CustomEvent('osprey-agent-pane'));
  }

  test('shows no badge while the pane is open, whatever waits', async () => {
    await boot();
    agentPane('open');
    chatEvents.emitChat('attention', { waiting: 3 });

    expect(agentBadge().hidden).toBe(true);
    expect(agentBadge().textContent).toBe('');
    expect(agentRow().dataset.tip).toBe('Sessions');
    expect(agentRow().hasAttribute('aria-description')).toBe(false);
  });

  test('shows the waiting count while the pane is closed', async () => {
    await boot();
    chatEvents.emitChat('attention', { waiting: 2 });
    agentPane('closed');

    expect(agentBadge().hidden).toBe(false);
    expect(agentBadge().textContent).toBe('2');
    expect(agentRow().dataset.tip).toBe('Sessions · 2 waiting — click to open');
    expect(agentRow().getAttribute('aria-description')).toBe(
      'The Agent pane is closed and 2 cards wait on you; activate to open it'
    );

    chatEvents.emitChat('attention', { waiting: 1 });
    expect(agentBadge().textContent).toBe('1');
    expect(agentRow().getAttribute('aria-description')).toBe(
      'The Agent pane is closed and 1 card waits on you; activate to open it'
    );
  });

  test('a closed pane with nothing waiting says closed and shows no badge', async () => {
    await boot();
    agentPane('closed');
    chatEvents.emitChat('attention', { waiting: 4 });
    chatEvents.emitChat('attention', { waiting: 0 });

    expect(agentBadge().hidden).toBe(true);
    expect(agentRow().dataset.tip).toBe('Sessions · Agent pane closed — click to open');
    expect(agentRow().getAttribute('aria-description')).toBe(
      'The Agent pane is closed; activate to open it'
    );
  });

  test('opening the pane again takes the badge down', async () => {
    await boot();
    agentPane('closed');
    chatEvents.emitChat('attention', { waiting: 2 });
    expect(agentBadge().hidden).toBe(false);

    agentPane('open');
    expect(agentBadge().hidden).toBe(true);
    expect(agentRow().dataset.tip).toBe('Sessions');
    expect(agentRow().hasAttribute('aria-description')).toBe(false);
  });

  test('its tip names the ⌥-click that expands or collapses every session (Alt+click elsewhere)', async () => {
    setPlatform('MacIntel');
    await boot();
    agentPane('open');
    expect(agentRow().dataset.tipDetail).toBe(
      'Click to fold the session list · ⌥-click to expand or collapse every session'
    );
    agentPane('closed');
    expect(agentRow().dataset.tipDetail).toBe(
      'Click to open the Agent pane · ⌥-click to expand or collapse every session'
    );
    setPlatform('Linux x86_64');
    agentPane('open');
    expect(agentRow().dataset.tipDetail).toBe(
      'Click to fold the session list · Alt+click to expand or collapse every session'
    );
  });

  test('a re-render of the rows repaints the row', async () => {
    await boot();
    document.documentElement.setAttribute('data-agent', 'closed');
    chatEvents.emitChat('attention', { waiting: 2 });
    // panel-rail.js re-labels the row on every render.
    agentRow().dataset.tip = 'Sessions';
    document.dispatchEvent(new CustomEvent('osprey-sidebar-rendered'));
    expect(agentRow().dataset.tip).toBe('Sessions · 2 waiting — click to open');
  });
});

/* ---- fades --------------------------------------------------------------- */

describe('fades', () => {
  /**
   * Give a scroller the scroll metrics happy-dom does not lay out.
   * @param {HTMLElement} el
   * @param {{scrollHeight: number, clientHeight: number, scrollTop: number}} metrics
   */
  function scrollMetrics(el, metrics) {
    for (const [key, value] of Object.entries(metrics)) {
      Object.defineProperty(el, key, { value, configurable: true, writable: true });
    }
  }

  test('a group that scrolls fades at each edge with more to show', async () => {
    await boot();
    const workspace = qs(document, '#panel-rail > [data-group="workspace"]');

    scrollMetrics(workspace, { scrollHeight: 400, clientHeight: 200, scrollTop: 0 });
    workspace.dispatchEvent(new Event('scroll'));
    expect(workspace.classList.contains('fade')).toBe(true);
    expect(workspace.classList.contains('fade-top')).toBe(false);

    scrollMetrics(workspace, { scrollHeight: 400, clientHeight: 200, scrollTop: 200 });
    workspace.dispatchEvent(new Event('scroll'));
    expect(workspace.classList.contains('fade')).toBe(false);
    expect(workspace.classList.contains('fade-top')).toBe(true);
  });

  test('a group a later render adds fades too', async () => {
    await boot();
    const added = document.createElement('div');
    added.className = 'sidebar-group';
    added.dataset.group = 'g-0123abcd';
    byId('panel-rail').append(added);
    document.dispatchEvent(new CustomEvent('osprey-sidebar-rendered'));

    scrollMetrics(added, { scrollHeight: 300, clientHeight: 100, scrollTop: 0 });
    added.dispatchEvent(new Event('scroll'));
    expect(added.classList.contains('fade')).toBe(true);
  });
});
