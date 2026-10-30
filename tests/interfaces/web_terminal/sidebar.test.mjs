// @ts-check
/**
 * The sidebar shell (sidebar.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar.test.mjs
 *
 * What the shell adds to the rows index.html ships and panel-rail.js renders:
 * the glyphs (empty `svg[data-icon]` placeholders filled in place), the
 * platform's key hints, the labels toggle (collapsible.js, under the scoped
 * `osprey-sidebar` key sidebar-boot.js reads), the Artifacts and Script runs
 * disclosures (kept under the scoped `osprey-sidebar-open` key), the collapsed
 * sidebar's Artifacts flyout, and the Agent row's waiting badge.
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

/** index.html's `nav#sidebar` and the body-level Artifacts flyout. */
const SIDEBAR_HTML = `
    <nav class="sidebar" id="sidebar" aria-label="Panels">
      <div class="sidebar-head">
        <a class="sidebar-brand" href="/" aria-label="OSPREY">
          <svg class="sidebar-brand-mark" viewBox="0 0 16 8" shape-rendering="crispEdges" aria-hidden="true" focusable="false"><path fill="currentColor" d="M0 0h2v1h-2zM14 0h2v1h-2zM1 1h3v1h-3zM12 1h3v1h-3zM2 2h4v1h-4zM10 2h4v1h-4zM3 3h4v1h-4zM9 3h4v1h-4zM4 4h8v1h-8zM6 5h4v1h-4zM7 6h2v1h-2zM6 7h1v1h-1zM9 7h1v1h-1z"/></svg>
          <span class="sidebar-brand-word">OSPREY</span>
        </a>
        <button class="osprey-icon-btn" id="sidebar-toggle" type="button"
                aria-label="Hide sidebar labels" aria-pressed="false" data-tip="Show labels">
          <svg class="osprey-ic" data-icon="panel-left" aria-hidden="true"></svg>
        </button>
      </div>

      <div class="sidebar-target" id="sidebar-target"></div>

      <button class="osprey-search" id="command-palette-btn" type="button"
              aria-label="Open command palette" data-tip="Search"
              title="Search settings, panels &amp; actions">
        <svg class="osprey-ic" data-icon="search" aria-hidden="true"></svg>
        <span class="sidebar-search-label">Search</span>
        <kbd class="osprey-kbd" data-chord="K" aria-hidden="true">&#8984;K</kbd>
      </button>

      <div class="sidebar-sections" id="panel-rail" role="list">
        <div class="sidebar-group sidebar-scroll" id="sidebar-workspace" data-group="workspace"
             role="listitem" aria-labelledby="sidebar-workspace-label">
          <div class="osprey-section-label" id="sidebar-workspace-label">Workspace</div>

          <button class="osprey-nav-row panel-rail-button disabled" id="sidebar-agent-row"
                  data-panel-id="terminal" type="button" aria-label="Agent" title="Agent" data-tip="Agent">
            <svg class="osprey-ic osprey-nav-icon" data-icon="agent" aria-hidden="true"></svg>
            <span class="osprey-nav-label">Agent</span>
            <span class="osprey-dot sidebar-agent-dot" data-agent-dot aria-hidden="true"></span>
            <span class="sidebar-open-hint" aria-hidden="true">Open</span>
            <span class="osprey-nav-badge" data-agent-badge hidden></span>
          </button>

          <div class="sidebar-row" id="sidebar-artifacts-row">
            <button class="osprey-nav-row parent panel-rail-button disabled" data-panel-id="artifacts"
                    type="button" aria-expanded="true" aria-controls="sidebar-artifacts-kids"
                    aria-label="Artifacts" title="Artifacts" data-tip="Artifacts">
              <svg class="osprey-ic osprey-nav-icon" data-icon="layers" aria-hidden="true"></svg>
              <span class="osprey-nav-label">Artifacts</span>
              <span class="osprey-nav-badge" data-art-count hidden></span>
              <svg class="osprey-ic osprey-nav-disc" data-disc data-icon="chevron-right" aria-hidden="true"></svg>
            </button>
            <div class="sidebar-row-actions">
              <button class="osprey-mini" type="button" data-art-filter-toggle title="Filter"
                      aria-label="Filter artifacts" aria-pressed="false"><svg class="osprey-ic" data-icon="search" aria-hidden="true"></svg></button>
              <button class="osprey-mini" type="button" data-art-menu title="More"
                      aria-label="Artifact options" aria-haspopup="menu" aria-expanded="false"
                      aria-controls="sidebar-artifacts-menu"><svg class="osprey-ic" data-icon="more" aria-hidden="true"></svg></button>
            </div>
            <div class="sidebar-kids" id="sidebar-artifacts-kids">
              <label class="sidebar-kids-filter" data-art-filter hidden>
                <svg class="osprey-ic" data-icon="search" aria-hidden="true"></svg>
                <input type="search" placeholder="Filter by title" aria-label="Filter artifacts by title">
              </label>
              <div class="sidebar-kids-list" id="sidebar-artifacts-list" role="listbox" aria-label="Artifacts"></div>
            </div>
          </div>

          <div class="sidebar-row" id="sidebar-runs-row">
            <button class="osprey-nav-row parent" id="sidebar-runs-toggle" type="button"
                    aria-expanded="false" aria-controls="sidebar-runs-kids" data-tip="Script runs">
              <svg class="osprey-ic osprey-nav-icon" data-icon="bash" aria-hidden="true"></svg>
              <span class="osprey-nav-label">Script runs</span>
              <span class="osprey-nav-badge" data-runs-count hidden></span>
              <svg class="osprey-ic osprey-nav-disc" data-disc data-icon="chevron-right" aria-hidden="true"></svg>
            </button>
            <div class="sidebar-kids" id="sidebar-runs-kids" hidden>
              <div class="sidebar-kids-list" id="sidebar-runs-list" role="listbox" aria-label="Script runs"></div>
            </div>
          </div>
        </div>

        <div class="sidebar-group" data-group="tools" role="listitem" aria-labelledby="sidebar-tools-label">
          <div class="osprey-section-label" id="sidebar-tools-label">Tools</div>
          <button class="osprey-nav-row add" id="panel-add-btn" type="button" aria-label="Add panel"
                  aria-haspopup="dialog" aria-expanded="false" aria-controls="panel-add-menu" data-tip="Add panel">
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

  <div class="osprey-popover sidebar-flyout" id="sidebar-artifacts-flyout" role="dialog" aria-label="Artifacts" hidden>
    <div class="sidebar-flyout-head">
      <b>Artifacts</b>
      <span class="osprey-count" data-art-count></span>
      <div class="sidebar-row-actions">
        <button class="osprey-mini" type="button" data-art-filter-toggle title="Filter"
                aria-label="Filter artifacts" aria-pressed="false"><svg class="osprey-ic" data-icon="search" aria-hidden="true"></svg></button>
        <button class="osprey-mini" type="button" data-art-menu title="More"
                aria-label="Artifact options" aria-haspopup="menu" aria-expanded="false"
                aria-controls="sidebar-artifacts-menu"><svg class="osprey-ic" data-icon="more" aria-hidden="true"></svg></button>
      </div>
    </div>
    <div class="sidebar-kids">
      <label class="sidebar-kids-filter" data-art-filter hidden>
        <svg class="osprey-ic" data-icon="search" aria-hidden="true"></svg>
        <input type="search" placeholder="Filter by title" aria-label="Filter artifacts by title">
      </label>
      <div class="sidebar-kids-list" id="sidebar-artifacts-flyout-list" role="listbox" aria-label="Artifacts"></div>
    </div>
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

const artifactsRow = () => qs(document, '#sidebar-artifacts-row > .osprey-nav-row');
const artifactsKids = () => byId('sidebar-artifacts-kids');
const runsToggle = () => byId('sidebar-runs-toggle');
const runsKids = () => byId('sidebar-runs-kids');
const flyout = () => byId('sidebar-artifacts-flyout');
const agentRow = () => byId('sidebar-agent-row');
const agentBadge = () => qs(agentRow(), '[data-agent-badge]');

/** Click a row's chevron, which is an svg (no `.click()` of its own). @param {HTMLElement} row */
function clickDisc(row) {
  qs(row, '[data-disc]').dispatchEvent(
    new MouseEvent('click', { bubbles: true, cancelable: true })
  );
}

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
  test('a first visit opens Artifacts and leaves Script runs closed', async () => {
    await load();
    // Paint the opposite first, so the defaults are the module's answer.
    artifactsKids().hidden = true;
    artifactsRow().setAttribute('aria-expanded', 'false');
    runsKids().hidden = false;
    runsToggle().setAttribute('aria-expanded', 'true');

    sidebar.initSidebar();

    expect(artifactsKids().hidden).toBe(false);
    expect(artifactsRow().getAttribute('aria-expanded')).toBe('true');
    expect(artifactsRow().getAttribute('aria-controls')).toBe('sidebar-artifacts-kids');
    expect(runsKids().hidden).toBe(true);
    expect(runsToggle().getAttribute('aria-expanded')).toBe('false');
    expect(runsToggle().getAttribute('aria-controls')).toBe('sidebar-runs-kids');
  });

  test('the open lists persist per user under the scoped key', async () => {
    document.documentElement.setAttribute(SCOPE_ATTR, 'alice');
    await boot();
    const key = `${sidebar.OPEN_LISTS_STORAGE_KEY}--alice`;

    runsToggle().click();
    expect(runsKids().hidden).toBe(false);
    expect(storedOpen(key)).toEqual(['artifacts', 'runs']);
    expect(window.localStorage.getItem(sidebar.OPEN_LISTS_STORAGE_KEY)).toBeNull();

    clickDisc(artifactsRow());
    expect(storedOpen(key)).toEqual(['runs']);

    // The next page load paints what was stored.
    await boot();
    expect(artifactsKids().hidden).toBe(true);
    expect(artifactsRow().getAttribute('aria-expanded')).toBe('false');
    expect(runsKids().hidden).toBe(false);
    expect(runsToggle().getAttribute('aria-expanded')).toBe('true');
  });

  test('another user on the same browser keeps the defaults', async () => {
    window.localStorage.setItem('osprey-sidebar-open--alice', JSON.stringify(['runs']));
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
    await boot();
    expect(artifactsKids().hidden).toBe(false);
    expect(runsKids().hidden).toBe(true);
  });

  test('an unreadable stored value falls back to the defaults', async () => {
    for (const raw of ['not json', '{"runs": true}']) {
      window.localStorage.setItem('osprey-sidebar-open', raw);
      await boot();
      expect(artifactsKids().hidden).toBe(false);
      expect(runsKids().hidden).toBe(true);
    }
  });

  test('a stored list keeps only the rows it knows', async () => {
    window.localStorage.setItem('osprey-sidebar-open', '["nope", 3]');
    await boot();
    expect(artifactsKids().hidden).toBe(true);
    expect(runsKids().hidden).toBe(true);
  });

  test('a click on the Artifacts row opens its list and never closes it', async () => {
    await boot();
    clickDisc(artifactsRow());
    expect(artifactsKids().hidden).toBe(true);

    artifactsRow().click();
    expect(artifactsKids().hidden).toBe(false);
    expect(artifactsRow().getAttribute('aria-expanded')).toBe('true');

    artifactsRow().click();
    expect(artifactsKids().hidden).toBe(false);
    expect(storedOpen('osprey-sidebar-open')).toEqual(['artifacts']);
  });

  test("the chevron toggles the list only, never the row's own click", async () => {
    await boot();
    const activate = vi.fn();
    // panel-rail.js's activation listens on the same row.
    artifactsRow().addEventListener('click', activate);

    clickDisc(artifactsRow());
    expect(artifactsKids().hidden).toBe(true);
    expect(artifactsRow().getAttribute('aria-expanded')).toBe('false');

    clickDisc(artifactsRow());
    expect(artifactsKids().hidden).toBe(false);
    expect(artifactsRow().getAttribute('aria-expanded')).toBe('true');

    expect(activate).not.toHaveBeenCalled();
  });

  test('ArrowRight opens and ArrowLeft closes a list from the keyboard', async () => {
    await boot();

    const left = press(artifactsRow(), 'ArrowLeft');
    expect(left.defaultPrevented).toBe(true);
    expect(artifactsKids().hidden).toBe(true);
    press(artifactsRow(), 'ArrowRight');
    expect(artifactsKids().hidden).toBe(false);

    press(runsToggle(), 'ArrowRight');
    expect(runsKids().hidden).toBe(false);
    expect(runsToggle().getAttribute('aria-expanded')).toBe('true');
    press(runsToggle(), 'ArrowLeft');
    expect(runsKids().hidden).toBe(true);

    // Other keys are left to the page.
    expect(press(runsToggle(), 'ArrowDown').defaultPrevented).toBe(false);
  });

  test('a click on Script runs toggles its list', async () => {
    await boot();
    runsToggle().click();
    expect(runsKids().hidden).toBe(false);
    runsToggle().click();
    expect(runsKids().hidden).toBe(true);
  });
});

/* ---- collapsed ----------------------------------------------------------- */

describe('collapsed', () => {
  test('Script runs, which has no flyout, does nothing', async () => {
    await boot();
    collapse();
    const stored = window.localStorage.getItem('osprey-sidebar-open');

    runsToggle().click();
    clickDisc(runsToggle());
    press(runsToggle(), 'ArrowRight');

    expect(runsKids().hidden).toBe(true);
    expect(runsToggle().getAttribute('aria-expanded')).toBe('false');
    expect(window.localStorage.getItem('osprey-sidebar-open')).toBe(stored);
  });

  test('the arrow keys leave the Artifacts list alone', async () => {
    await boot();
    collapse();
    const left = press(artifactsRow(), 'ArrowLeft');
    expect(left.defaultPrevented).toBe(false);
    expect(artifactsKids().hidden).toBe(false);
  });

  test('the Artifacts row opens the flyout beside the sidebar and closes it again', async () => {
    await boot();
    collapse();

    artifactsRow().click();
    expect(flyout().hidden).toBe(false);
    expect(artifactsRow().getAttribute('aria-controls')).toBe('sidebar-artifacts-flyout');
    expect(artifactsRow().getAttribute('aria-expanded')).toBe('true');
    // The flyout is the list, not a change to it.
    expect(storedOpen('osprey-sidebar-open')).toBeNull();

    artifactsRow().click();
    expect(flyout().hidden).toBe(true);
    expect(artifactsRow().getAttribute('aria-expanded')).toBe('false');
  });

  test('back to labels closes the flyout and points the row at its list again', async () => {
    await boot();
    collapse();
    artifactsRow().click();
    expect(flyout().hidden).toBe(false);

    collapse();

    expect(document.documentElement.getAttribute('data-sidebar')).toBe('expanded');
    expect(flyout().hidden).toBe(true);
    expect(artifactsRow().getAttribute('aria-controls')).toBe('sidebar-artifacts-kids');
    expect(artifactsRow().getAttribute('aria-expanded')).toBe('true');
    expect(artifactsKids().hidden).toBe(false);
  });
});

/* ---- the Agent row ------------------------------------------------------- */

describe('the Agent row', () => {
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
    expect(agentRow().dataset.tip).toBe('Agent');
    expect(agentRow().hasAttribute('aria-description')).toBe(false);
  });

  test('shows the waiting count while the pane is closed', async () => {
    await boot();
    chatEvents.emitChat('attention', { waiting: 2 });
    agentPane('closed');

    expect(agentBadge().hidden).toBe(false);
    expect(agentBadge().textContent).toBe('2');
    expect(agentRow().dataset.tip).toBe('Agent · 2 waiting — click to open');
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
    expect(agentRow().dataset.tip).toBe('Agent · closed — click to open');
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
    expect(agentRow().dataset.tip).toBe('Agent');
    expect(agentRow().hasAttribute('aria-description')).toBe(false);
  });

  test('a re-render of the rows repaints the row', async () => {
    await boot();
    document.documentElement.setAttribute('data-agent', 'closed');
    chatEvents.emitChat('attention', { waiting: 2 });
    // panel-rail.js re-labels the row on every render.
    agentRow().dataset.tip = 'Agent';
    document.dispatchEvent(new CustomEvent('osprey-sidebar-rendered'));
    expect(agentRow().dataset.tip).toBe('Agent · 2 waiting — click to open');
  });
});
