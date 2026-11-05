// @vitest-environment jsdom
/**
 * Contract tests for the tile bar's shared controls (tile-bar-controls.js):
 * the configured ids on `<html data-tile-bar-controls>` become
 * `.osprey-icon-btn[data-control]` buttons behind the hairline; `ask_agent`
 * opens the Agent pane and raises `osprey-agent-compose` with the panel's
 * deep link, and follows the Agent surface; `reload` reloads the frame;
 * `help` opens the docs page and is absent without one; folded controls are
 * offered as "This panel" rows. Run:
 *   npx vitest run tests/interfaces/web_terminal/tile-bar-controls.test.mjs
 */
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const { openTerminalPanel, panelLink, reloadPanel, helpUrl, openHelp } = vi.hoisted(() => ({
  openTerminalPanel: vi.fn(() => true),
  panelLink: vi.fn(() => 'http://localhost/panel/lattice/?model=sr#tunes'),
  reloadPanel: vi.fn(() => true),
  helpUrl: vi.fn(() => 'https://docs.example/how-to/web-terminal/lattice-dashboard.html'),
  openHelp: vi.fn(() => true),
}));

// The dock (dockview itself) is out of scope: the control only has to open the
// Agent pane through the layout holder's verb.
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  openTerminalPanel,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-links.js', () => ({
  panelLink,
  reloadPanel,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-help.js', () => ({
  helpUrl,
  openHelp,
}));

const {
  AGENT_COMPOSE_EVENT,
  ASK_AGENT_TERMINAL_TITLE,
  buildSharedControls,
  configuredControls,
  foldedControlRows,
  unregisterSharedControls,
} = await import(`${JS}/tile-bar-controls.js`);
const { showChatSurface, showTerminalSurface } = await import(`${JS}/agent-surface.js`);

/** @param {string} value */
function configure(value) {
  document.documentElement.setAttribute('data-tile-bar-controls', value);
}

/** @param {HTMLElement} el @returns {string[]} */
const controlIds = (el) =>
  [...el.querySelectorAll('[data-control]')].map(
    (b) => /** @type {HTMLElement} */ (b).dataset.control
  );

beforeEach(() => {
  configure('ask_agent reload help');
  document.documentElement.removeAttribute('data-agent-surface');
  document.body.replaceChildren();
  vi.clearAllMocks();
});

afterEach(() => {
  showChatSurface();
});

describe('configuredControls', () => {
  test('reads the ids in order, drops unknown ones and repeats, and reads an empty list as none', () => {
    configure('help  reload ask_agent reload bogus');
    expect(configuredControls()).toEqual(['help', 'reload', 'ask_agent']);
    configure('');
    expect(configuredControls()).toEqual([]);
    document.documentElement.removeAttribute('data-tile-bar-controls');
    expect(configuredControls()).toEqual([]);
  });
});

describe('buildSharedControls', () => {
  test('the hairline first, then one icon button per configured id with tooltip and name', () => {
    const el = buildSharedControls('lattice', () => 'Lattice');
    document.body.appendChild(el);

    expect(el.className).toBe('tile-tab-controls');
    const divider = /** @type {HTMLElement} */ (el.firstElementChild);
    expect(divider.className).toBe('tile-tab-divider');
    expect(divider.getAttribute('aria-hidden')).toBe('true');
    expect(divider.hidden).toBe(true); // the ladder shows it once both sides have a control
    expect(controlIds(el)).toEqual(['ask_agent', 'reload', 'help']);

    const [ask, reload, help] = /** @type {HTMLButtonElement[]} */ ([
      ...el.querySelectorAll('button.osprey-icon-btn[data-control]'),
    ]);
    expect(ask.title).toBe('Ask the agent about this panel');
    expect(ask.getAttribute('aria-label')).toBe('Ask the agent about this panel');
    expect(ask.querySelector('svg.osprey-ic')).toBeTruthy();
    expect(reload.title).toBe('Reload panel');
    expect(help.title).toBe('Help for this panel');
    expect(help.hidden).toBe(false);
  });

  test('an empty configuration builds the hairline and nothing else', () => {
    configure('');
    const el = buildSharedControls('lattice', () => 'Lattice');
    expect(controlIds(el)).toEqual([]);
    expect(el.children.length).toBe(1);
  });

  test('ask_agent opens the Agent pane and hands the composer a line with the deep link, unsent', () => {
    const el = buildSharedControls('lattice', () => 'Lattice');
    document.body.appendChild(el);
    const composer = document.createElement('textarea');
    const container = document.createElement('div');
    container.id = 'operator-container';
    container.appendChild(composer);
    document.body.appendChild(container);
    const heard = vi.fn();
    document.addEventListener(AGENT_COMPOSE_EVENT, heard);

    /** @type {HTMLButtonElement} */ (el.querySelector('[data-control="ask_agent"]')).click();

    expect(openTerminalPanel).toHaveBeenCalledTimes(1);
    expect(document.activeElement).toBe(composer);
    expect(panelLink).toHaveBeenCalledWith('lattice');
    expect(heard).toHaveBeenCalledTimes(1);
    expect(heard.mock.calls[0][0].detail).toEqual({
      text: 'About the Lattice panel (http://localhost/panel/lattice/?model=sr#tunes): ',
    });
  });

  test('ask_agent reads the title at click time, and leaves the link out when the panel has none', () => {
    let title = 'Channels';
    const el = buildSharedControls('channel-finder', () => title);
    panelLink.mockReturnValueOnce(null);
    const heard = vi.fn();
    document.addEventListener(AGENT_COMPOSE_EVENT, heard);

    title = 'Channel Finder';
    /** @type {HTMLButtonElement} */ (el.querySelector('[data-control="ask_agent"]')).click();

    expect(heard.mock.calls[0][0].detail.text).toBe('About the Channel Finder panel: ');
  });

  test('ask_agent is disabled, with the reason, while the Agent pane shows the terminal', () => {
    const el = buildSharedControls('lattice', () => 'Lattice');
    const ask = /** @type {HTMLButtonElement} */ (el.querySelector('[data-control="ask_agent"]'));
    expect(ask.disabled).toBe(false);

    showTerminalSurface();
    expect(ask.disabled).toBe(true);
    expect(ask.title).toBe(ASK_AGENT_TERMINAL_TITLE);

    showChatSurface();
    expect(ask.disabled).toBe(false);
    expect(ask.title).toBe('Ask the agent about this panel');
  });

  test('a tile built while the terminal shows starts disabled', () => {
    showTerminalSurface();
    const el = buildSharedControls('lattice', () => 'Lattice');
    const ask = /** @type {HTMLButtonElement} */ (el.querySelector('[data-control="ask_agent"]'));
    expect(ask.disabled).toBe(true);
    expect(ask.title).toBe(ASK_AGENT_TERMINAL_TITLE);
  });

  test('reload reloads the panel frame', () => {
    const el = buildSharedControls('lattice', () => 'Lattice');
    /** @type {HTMLButtonElement} */ (el.querySelector('[data-control="reload"]')).click();
    expect(reloadPanel).toHaveBeenCalledWith('lattice');
  });

  test('help opens the docs page, and is hidden for a panel without one', () => {
    const el = buildSharedControls('lattice', () => 'Lattice');
    /** @type {HTMLButtonElement} */ (el.querySelector('[data-control="help"]')).click();
    expect(openHelp).toHaveBeenCalledWith('lattice');

    helpUrl.mockReturnValueOnce(null);
    const bare = buildSharedControls('custom', () => 'Custom');
    expect(
      /** @type {HTMLButtonElement} */ (bare.querySelector('[data-control="help"]')).hidden
    ).toBe(true);
  });
});

describe('foldedControlRows', () => {
  test('offers the folded controls as rows that run the button, and nothing for a hidden one', () => {
    const el = buildSharedControls('lattice', () => 'Lattice');
    const [ask, reload, help] = /** @type {HTMLButtonElement[]} */ ([
      ...el.querySelectorAll('[data-control]'),
    ]);
    expect(foldedControlRows('lattice')).toEqual([]);

    reload.classList.add('contrib-hidden');
    help.classList.add('contrib-hidden');
    help.hidden = true;
    const rows = foldedControlRows('lattice');
    expect(rows.map((r) => [r.label, r.disabled])).toEqual([['Reload panel', false]]);
    rows[0].run?.();
    expect(reloadPanel).toHaveBeenCalledWith('lattice');

    // A disabled folded control stays a disabled row, under the control's name.
    showTerminalSurface();
    ask.classList.add('contrib-hidden');
    expect(foldedControlRows('lattice').map((r) => [r.label, r.disabled])).toEqual([
      ['Ask the agent about this panel', true],
      ['Reload panel', false],
    ]);
  });

  test('a disposed tile offers no rows; a rebuilt tile replaces the earlier registration', () => {
    const first = buildSharedControls('lattice', () => 'Lattice');
    const second = buildSharedControls('lattice', () => 'Lattice');
    /** @type {HTMLElement} */ (second.querySelector('[data-control="reload"]')).classList.add(
      'contrib-hidden'
    );
    // The rebuild registers before the old tile disposes; the old dispose is a no-op.
    unregisterSharedControls('lattice', first);
    expect(foldedControlRows('lattice').map((r) => r.label)).toEqual(['Reload panel']);

    unregisterSharedControls('lattice', second);
    expect(foldedControlRows('lattice')).toEqual([]);
  });
});
