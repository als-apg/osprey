// @ts-check
/**
 * OSPREY Web Terminal — JupyterLab, framed as the Jupyter tile, takes the
 * hub's theme.
 *
 * Every other panel follows the hub's theme through theme-manager.js: the
 * `?theme=` its frame is loaded with, then each `osprey-theme-change` the hub
 * posts. JupyterLab is not a design-system page. It themes itself, from its own
 * settings, so in a dark hub the tile stayed the light Lab the sidecar pinned
 * at start (`jupyter_sidecar.lab_theme_override`). This module carries the
 * hub's theme across from inside the Lab page: the frame's `?theme=` once Lab
 * has started, then every `osprey-theme-change` message.
 *
 * - **By mode.** The hub's themes are design-system palettes Lab has no copy
 *   of; what carries over is their light/dark side, read off the hub's own
 *   registry (`tokens.js`). Every dark theme becomes `JupyterLab Dark`, every
 *   light one `JupyterLab Light`: the two themes Lab ships, and the two names
 *   `jupyter_sidecar._THEME_NAMES` pins.
 * - **Through Lab's own command.** The sidecar's `page_config.json` sets
 *   `exposeAppInBrowser` (`jupyter_sidecar.lab_page_config`), so Lab publishes
 *   its application as `window.jupyterapp`, and the switch is
 *   `apputils:change-theme` from `@jupyterlab/apputils-extension:themes` — the
 *   command Lab's own *Settings → Theme* menu runs.
 * - **Only in the hub.** The gate is `embedded=true` on the page's address,
 *   which the hub's `buildEmbedSrc` stamps on every panel frame. A Lab page
 *   opened at its own address, or popped out of the hub, carries none and
 *   keeps the theme Lab chose for itself.
 *
 * The command stores what it switches to, exactly as Lab's menu does: the
 * theme setting, in Lab's settings on the shared root. So a Lab page opened on
 * its own starts in the theme the tile followed last, and a pick made there
 * holds until the tile follows the hub again. The sidecar's pin stays what Lab
 * paints while it has no stored theme.
 *
 * **When Lab can take a switch.** Lab sets `window.jupyterapp` right after it
 * calls `start()`, long before its theme manager can answer, and until then it
 * paints its own theme (the stored one, else the pin). The ready signal is
 * the one Lab publishes itself, `<body data-jp-theme-name>`: present once the
 * theme manager holds its settings and has loaded a theme. Then `started`
 * (every startup plugin is active, the theme plugin among them) and the
 * command itself; a Lab without the app or the command keeps its theme, and
 * the console says why. `restored` is not waited for: it holds until Lab has
 * reopened the whole workspace, and the theme plugin does not need that.
 *
 * **One switch at a time, each settled.** A switch is a settings write, then a
 * stylesheet load, and Lab drops a request for the theme it has LOADED. Read
 * against a half-finished switch, a request (dark, light, dark again in quick
 * succession) would be dropped as a repeat and leave Lab on the middle one. So
 * requests collapse to the latest, and each switch waits for Lab to publish
 * its theme before the next request is read.
 *
 * Booted by the page's entry module, `control-target-lab-bar.js`, which the
 * panel proxy injects into every Lab application page.
 */

// Not theme-manager.js, and not frame-params.js for `embedded`: both import
// `/design-system/js/storage-scope.js` themselves, and on this page only the
// modules in this directory have that specifier mapped onto a per-user mount
// (`_control_target_bar_markup` in routes/proxy.py). tokens.js imports nothing.
import { THEMES } from '/design-system/js/tokens.js';

/** Lab's theme switch: the command its *Settings → Theme* menu runs. */
const CHANGE_THEME = 'apputils:change-theme';

/** Lab's *Synchronize with System Settings* toggle, from the same plugin. */
const ADAPTIVE_THEME = 'apputils:adaptive-theme';

/**
 * The Lab theme for each side of the hub's light/dark. Keep equal to
 * `_THEME_NAMES` in `jupyter_sidecar.py`, which pins the same two at start.
 * @type {Readonly<Record<string, string>>}
 */
const LAB_THEMES = Object.freeze({ dark: 'JupyterLab Dark', light: 'JupyterLab Light' });

/** The message the hub posts to every panel frame (theme-manager.js, panel-iframe-sync.js). */
const MESSAGE_TYPE = 'osprey-theme-change';

/** Where Lab publishes the theme it has loaded. */
const THEME_NAME_ATTR = 'data-jp-theme-name';

/**
 * How long one switch may take to land before the next request is read
 * anyway: a theme whose stylesheet never loads must not hold the hub's later
 * picks behind it.
 */
const SETTLE_MS = 5000;

/**
 * The part of JupyterLab's application object (`window.jupyterapp`) this
 * module drives.
 * @typedef {object} LabApp
 * @property {Promise<void>} started resolves once every startup plugin is active
 * @property {{
 *   hasCommand: (id: string) => boolean,
 *   isToggled: (id: string) => boolean,
 *   execute: (id: string, args?: object) => Promise<unknown>,
 * }} commands
 */

/** The Lab theme the hub asked for last. @type {string|null} */
let wanted = null;

/** The switch loop, while one runs. @type {Promise<void>|null} */
let pump = null;

/** Whether this page follows the hub: set by init, cleared by teardown. */
let following = false;

/** One release per wait in progress, so teardown can end them. @type {Set<() => void>} */
const pendingWaits = new Set();

/**
 * The Lab theme a hub theme id stands for, by the id's mode in the hub's own
 * registry — so a theme family added there needs nothing here. `null` for an
 * id the registry does not list, which includes anything a message makes up.
 *
 * @param {unknown} themeId
 * @returns {string|null}
 */
export function labThemeFor(themeId) {
  const entry = THEMES.find((theme) => theme.id === themeId);
  return entry ? (LAB_THEMES[entry.mode] ?? null) : null;
}

/**
 * Start following the hub's theme, when the hub framed this page; a no-op
 * anywhere else, and on a second call.
 *
 * The address's `theme` is read here, before Lab boots: Lab's router rewrites
 * the address as documents open, and the query goes with it.
 *
 * @returns {boolean} whether this page follows the hub
 */
export function initLabThemeFollow() {
  if (following) return true;
  if (typeof document === 'undefined' || !document.body) return false;
  const params = new URLSearchParams(window.location.search);
  if (params.get('embedded') !== 'true') return false;

  following = true;
  window.addEventListener('message', onMessage);
  request(params.get('theme'));
  return true;
}

/** Stop following: drop the listener, end every wait, forget the request. */
export function teardownLabThemeFollow() {
  following = false;
  wanted = null;
  window.removeEventListener('message', onMessage);
  for (const release of [...pendingWaits]) release();
}

/**
 * The hub's theme broadcast. Taken by origin, the rule theme-manager.js's
 * follower applies on a same-origin page; `labThemeFor` then drops any id the
 * registry does not list before it can reach Lab.
 *
 * @param {MessageEvent} event
 */
function onMessage(event) {
  if (event.origin !== window.location.origin) return;
  const data = event.data;
  if (!data || data.type !== MESSAGE_TYPE) return;
  request(data.theme);
}

/**
 * Ask for the Lab theme of a hub theme id. The latest request wins; one loop
 * serves them all.
 *
 * @param {unknown} themeId
 */
function request(themeId) {
  const theme = labThemeFor(themeId);
  if (!following || theme === null) return;
  wanted = theme;
  pump ??= follow()
    .catch((err) => console.warn('OSPREY: JupyterLab did not take the hub theme:', err))
    .finally(() => {
      pump = null;
    });
}

/**
 * Switch Lab to `wanted`, then to each later `wanted`, one settled switch at a
 * time, once Lab can take one.
 *
 * @returns {Promise<void>}
 */
async function follow() {
  if (!(await whenLabTheme(Boolean)) || !following) return;
  const app = /** @type {LabApp | undefined} */ (Reflect.get(window, 'jupyterapp'));
  if (!app?.commands) {
    stop('JupyterLab does not publish window.jupyterapp (exposeAppInBrowser is off)');
    return;
  }
  await app.started;
  if (!following) return;
  if (!app.commands.hasCommand(CHANGE_THEME)) {
    stop(`JupyterLab has no ${CHANGE_THEME} command`);
    return;
  }

  /** @type {string|null} */
  let done = null;
  while (following && wanted !== null && wanted !== done) {
    const theme = /** @type {string} */ (wanted);
    done = theme;
    if (document.body.getAttribute(THEME_NAME_ATTR) === theme) continue;
    // With Lab's own "Synchronize with System Settings" on, Lab answers a
    // pick by turning that off and nothing more (its menu's rule), so the
    // pick is made a second time.
    const adaptive = app.commands.isToggled(ADAPTIVE_THEME);
    await app.commands.execute(CHANGE_THEME, { theme });
    if (adaptive) await app.commands.execute(CHANGE_THEME, { theme });
    await whenLabTheme((name) => name === theme, SETTLE_MS);
  }
}

/**
 * Give up on a Lab that cannot be driven: say why once, and stop listening.
 *
 * @param {string} why
 */
function stop(why) {
  console.warn(`OSPREY: the Jupyter tile keeps JupyterLab's own theme: ${why}.`);
  teardownLabThemeFollow();
}

/**
 * Resolve `true` once the theme Lab publishes satisfies `accept`; `false`
 * after `timeoutMs`, or when teardown comes first.
 *
 * @param {(name: string | null) => boolean} accept
 * @param {number} [timeoutMs] no limit when absent
 * @returns {Promise<boolean>}
 */
function whenLabTheme(accept, timeoutMs) {
  const body = document.body;
  if (accept(body.getAttribute(THEME_NAME_ATTR))) return Promise.resolve(true);
  return new Promise((resolve) => {
    /** @type {ReturnType<typeof setTimeout> | undefined} */
    let timer;
    const observer = new MutationObserver(() => {
      if (accept(body.getAttribute(THEME_NAME_ATTR))) end(true);
    });
    /** @param {boolean} met */
    const end = (met) => {
      observer.disconnect();
      clearTimeout(timer);
      pendingWaits.delete(release);
      resolve(met);
    };
    const release = () => end(false);
    pendingWaits.add(release);
    observer.observe(body, { attributes: true, attributeFilter: [THEME_NAME_ATTR] });
    if (timeoutMs !== undefined) timer = setTimeout(release, timeoutMs);
  });
}
