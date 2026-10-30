// @ts-check
/**
 * Unit tests for the Jupyter tile's theme follower (lab-theme-follow.js):
 *   npx vitest run tests/interfaces/web_terminal/lab-theme-follow.test.mjs
 *
 * JupyterLab is faked at the one seam the module drives, `window.jupyterapp`'s
 * `started` and `commands`, with the two rules of Lab's theme plugin
 * (`@jupyterlab/apputils-extension:themes`) the module is written against:
 * `apputils:change-theme` drops a request for the theme Lab has LOADED, and
 * with "Synchronize with System Settings" on it only turns that off. A switch
 * lands on `<body data-jp-theme-name>` a task later, the way Lab publishes a
 * theme once its stylesheet has loaded.
 *
 * The module does not boot itself (control-target-lab-bar.js, the page's
 * entry, starts it), so each test resets the module registry, imports a fresh
 * copy and calls init at the address it needs.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { THEMES } from '/design-system/js/tokens.js';

const MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/lab-theme-follow.js';

/** Lab's command ids, written out: a rename on either side must fail here. */
const CHANGE_THEME = 'apputils:change-theme';
const ADAPTIVE_THEME = 'apputils:adaptive-theme';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/lab-theme-follow.js')} */
let follow;

/* ---- harness ------------------------------------------------------------ */

/** The theme Lab has published as loaded. */
const loadedTheme = () => document.body.getAttribute('data-jp-theme-name');

/**
 * Lab's theme manager publishing a theme it has loaded.
 * @param {string} name
 */
function publish(name) {
  document.body.setAttribute('data-jp-theme-name', name);
}

/**
 * Put a fake JupyterLab on `window.jupyterapp`. Not started, and no theme
 * loaded, until the test says so.
 *
 * @param {{commands?: string[], adaptive?: boolean, manualLoads?: boolean}} [options]
 *   `manualLoads` holds every switch until `finishLoad()` lands it, oldest first.
 */
function installLab({
  commands = [CHANGE_THEME, ADAPTIVE_THEME],
  adaptive = false,
  manualLoads = false,
} = {}) {
  const state = { adaptive };
  /** @type {string[]} */
  const loading = [];
  /** @type {() => void} */
  let start = () => {};
  /** @type {Promise<void>} */
  const started = new Promise((resolve) => {
    start = () => resolve();
  });
  const execute = vi.fn(async (/** @type {string} */ id, /** @type {{theme: string}} */ args) => {
    if (id !== CHANGE_THEME) return;
    if (args.theme === loadedTheme()) return;
    if (state.adaptive) {
      state.adaptive = false;
      return;
    }
    await Promise.resolve(); // the settings write
    if (manualLoads) loading.push(args.theme);
    else setTimeout(() => publish(args.theme), 0); // the stylesheet load
  });
  Reflect.set(window, 'jupyterapp', {
    started,
    commands: {
      hasCommand: (/** @type {string} */ id) => commands.includes(id),
      isToggled: (/** @type {string} */ id) => id === ADAPTIVE_THEME && state.adaptive,
      execute,
    },
  });
  return {
    execute,
    state,
    start: () => start(),
    load: publish,
    finishLoad: () => {
      const next = loading.shift();
      if (next !== undefined) publish(next);
    },
  };
}

/**
 * The themes the fake was asked to switch to, in order.
 * @param {ReturnType<typeof installLab>} lab
 */
const asked = (lab) => lab.execute.mock.calls.map(([, args]) => args.theme);

/**
 * Load the Lab page at `search` and start the follower, as the entry does.
 * @param {string} search
 */
function openAt(search) {
  window.history.replaceState({}, '', `/panel/jupyter/lab${search}`);
  return follow.initLabThemeFollow();
}

/**
 * Post a theme message to this window, the way the hub posts one to the tile.
 * @param {unknown} theme
 * @param {{origin?: string, type?: string}} [options]
 */
function postTheme(theme, { origin = window.location.origin, type = 'osprey-theme-change' } = {}) {
  window.dispatchEvent(new MessageEvent('message', { origin, data: { type, theme } }));
}

/** Drain the microtask and timer queue the follower chains through. */
async function flush() {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0));
}

beforeEach(async () => {
  vi.resetModules();
  follow = await import(MODULE);
});

afterEach(() => {
  follow.teardownLabThemeFollow();
  Reflect.deleteProperty(window, 'jupyterapp');
  document.body.removeAttribute('data-jp-theme-name');
  window.history.replaceState({}, '', '/');
  vi.restoreAllMocks();
});

/* ---- the mapping -------------------------------------------------------- */

describe('labThemeFor', () => {
  test("maps every theme in the hub's registry by its mode", () => {
    for (const { id, mode } of THEMES) {
      expect(follow.labThemeFor(id)).toBe(mode === 'dark' ? 'JupyterLab Dark' : 'JupyterLab Light');
    }
  });

  test.each([
    ['dark', 'JupyterLab Dark'],
    ['high-contrast-dark', 'JupyterLab Dark'],
    ['desy-dark', 'JupyterLab Dark'],
    ['retro-dark', 'JupyterLab Dark'],
    ['light', 'JupyterLab Light'],
    ['high-contrast-light', 'JupyterLab Light'],
    ['desy-light', 'JupyterLab Light'],
    ['retro-light', 'JupyterLab Light'],
  ])('%s is %s', (id, lab) => {
    expect(follow.labThemeFor(id)).toBe(lab);
  });

  test('names no Lab theme for an id the registry does not list', () => {
    for (const id of [undefined, null, '', 'null', 'auto', 'DARK', 'JupyterLab Dark', 42, {}]) {
      expect(follow.labThemeFor(id)).toBeNull();
    }
  });
});

/* ---- framed by the hub -------------------------------------------------- */

describe('on a page the hub framed (embedded=true)', () => {
  test("switches to the address's theme once Lab has a theme and has started", async () => {
    const lab = installLab();
    expect(openAt('?embedded=true&theme=dark&mode=expert')).toBe(true);
    await flush();
    // Lab is still booting: there is no theme manager to answer yet.
    expect(lab.execute).not.toHaveBeenCalled();

    lab.load('JupyterLab Light'); // the theme the sidecar pinned
    await flush();
    expect(lab.execute).not.toHaveBeenCalled();

    lab.start();
    await vi.waitFor(() => expect(loadedTheme()).toBe('JupyterLab Dark'));
    expect(lab.execute.mock.calls).toEqual([[CHANGE_THEME, { theme: 'JupyterLab Dark' }]]);
  });

  test('follows each theme the hub posts, by mode', async () => {
    const lab = installLab();
    lab.load('JupyterLab Light');
    lab.start();
    openAt('?embedded=true&theme=light');

    postTheme('high-contrast-dark');
    await vi.waitFor(() => expect(loadedTheme()).toBe('JupyterLab Dark'));
    postTheme('desy-light');
    await vi.waitFor(() => expect(loadedTheme()).toBe('JupyterLab Light'));

    expect(asked(lab)).toEqual(['JupyterLab Dark', 'JupyterLab Light']);
  });

  test('leaves Lab alone while it already wears the mode the hub asks for', async () => {
    const lab = installLab();
    lab.load('JupyterLab Dark');
    lab.start();
    openAt('?embedded=true&theme=dark');

    postTheme('retro-dark');
    postTheme('desy-dark');
    await flush();

    // No command, so no write to Lab's stored settings either.
    expect(lab.execute).not.toHaveBeenCalled();
  });

  test('drops a message from elsewhere, of another type, or naming no listed theme', async () => {
    const lab = installLab();
    lab.load('JupyterLab Light');
    lab.start();
    openAt('?embedded=true&theme=light');

    postTheme('dark', { origin: 'https://elsewhere.example' });
    postTheme('dark', { type: 'osprey-mode-change' });
    for (const theme of ['JupyterLab Dark', 'auto', '', undefined]) postTheme(theme);
    window.dispatchEvent(
      new MessageEvent('message', { origin: window.location.origin, data: null })
    );
    await flush();

    expect(lab.execute).not.toHaveBeenCalled();
    expect(loadedTheme()).toBe('JupyterLab Light');
  });

  test('reads the next request only once the switch before it has landed', async () => {
    // Asked for light while dark is still loading, Lab would compare against
    // the light it still has loaded, drop the request, and end on dark.
    const lab = installLab({ manualLoads: true });
    lab.load('JupyterLab Light');
    lab.start();
    openAt('?embedded=true&theme=light');

    postTheme('dark');
    await vi.waitFor(() => expect(lab.execute).toHaveBeenCalledTimes(1));
    postTheme('light');
    await flush();
    expect(lab.execute).toHaveBeenCalledTimes(1);

    lab.finishLoad(); // dark lands
    await vi.waitFor(() => expect(lab.execute).toHaveBeenCalledTimes(2));
    lab.finishLoad(); // light lands

    expect(loadedTheme()).toBe('JupyterLab Light');
    expect(asked(lab)).toEqual(['JupyterLab Dark', 'JupyterLab Light']);
  });

  test('collapses a burst of requests to the latest', async () => {
    const lab = installLab();
    lab.load('JupyterLab Light');
    lab.start();
    openAt('?embedded=true&theme=light');
    await flush();

    postTheme('dark');
    postTheme('light');
    postTheme('retro-dark');
    await vi.waitFor(() => expect(loadedTheme()).toBe('JupyterLab Dark'));

    expect(asked(lab)).toEqual(['JupyterLab Dark']);
  });

  test("turns Lab's own system-theme sync off first, then switches", async () => {
    const lab = installLab({ adaptive: true });
    lab.load('JupyterLab Light');
    lab.start();
    openAt('?embedded=true&theme=dark');

    await vi.waitFor(() => expect(loadedTheme()).toBe('JupyterLab Dark'));
    expect(lab.state.adaptive).toBe(false);
    expect(asked(lab)).toEqual(['JupyterLab Dark', 'JupyterLab Dark']);
  });

  test('reports a switch Lab refuses, and still takes the next request', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const lab = installLab();
    lab.execute.mockRejectedValueOnce(new Error('settings write refused'));
    lab.load('JupyterLab Light');
    lab.start();
    openAt('?embedded=true&theme=dark');

    await vi.waitFor(() => expect(warn).toHaveBeenCalledTimes(1));
    expect(loadedTheme()).toBe('JupyterLab Light');

    postTheme('dark');
    await vi.waitFor(() => expect(loadedTheme()).toBe('JupyterLab Dark'));
  });

  test("keeps Lab's theme when Lab has no theme command, says why once, and stops", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const lab = installLab({ commands: [] });
    lab.load('JupyterLab Light');
    lab.start();
    openAt('?embedded=true&theme=dark');

    await vi.waitFor(() => expect(warn).toHaveBeenCalledTimes(1));
    expect(String(warn.mock.calls[0][0])).toContain(CHANGE_THEME);

    postTheme('dark');
    await flush();
    expect(lab.execute).not.toHaveBeenCalled();
    expect(warn).toHaveBeenCalledTimes(1);
  });

  test("keeps Lab's theme when Lab does not publish its app, and says why", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    openAt('?embedded=true&theme=dark');

    publish('JupyterLab Light'); // Lab booted, with no window.jupyterapp
    await vi.waitFor(() => expect(warn).toHaveBeenCalledTimes(1));

    expect(String(warn.mock.calls[0][0])).toContain('exposeAppInBrowser');
    expect(loadedTheme()).toBe('JupyterLab Light');
  });

  test('teardown stops the follower, including a wait for Lab in progress', async () => {
    const lab = installLab();
    openAt('?embedded=true&theme=dark'); // Lab has no theme yet: the follower waits

    follow.teardownLabThemeFollow();
    lab.load('JupyterLab Light');
    lab.start();
    postTheme('dark');
    await flush();

    expect(lab.execute).not.toHaveBeenCalled();
  });
});

/* ---- on its own --------------------------------------------------------- */

describe('on a page opened on its own', () => {
  test.each(['?theme=dark', '?embedded=false&theme=dark', '?embedded=1&theme=dark'])(
    '%s: no switch from the address, and no listener',
    async (search) => {
      const lab = installLab();
      lab.load('JupyterLab Light');
      lab.start();

      expect(openAt(search)).toBe(false);
      postTheme('dark');
      await flush();

      expect(lab.execute).not.toHaveBeenCalled();
      expect(loadedTheme()).toBe('JupyterLab Light');
    }
  );
});
