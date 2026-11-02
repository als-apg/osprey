/**
 * scaffold-gallery.js's palette seams: listGalleryEntries (which tab shows
 * which file, labelled as its card) and revealArtifact (drawer open through
 * the gated openDrawerTab, then that file's Preview once the gallery loads).
 *
 *   npx vitest run tests/interfaces/web_terminal/scaffold/reveal.test.mjs
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

const ARTIFACTS = [
  { name: 'rules/safety', category: 'rules', description: 'Safety rules', status: 'framework' },
  {
    name: 'code-review/rules',
    category: 'code-review',
    description: "The code reviewer's instructions",
    status: 'framework',
    kind: 'data',
    read_only: true,
  },
  { name: 'hooks/approval', category: 'hooks', description: 'Approval hook', status: 'framework' },
  { name: 'mcp-json', category: 'config', description: 'MCP servers', status: 'framework' },
];

const fetchJSON = vi.fn(async (/** @type {string} */ url) => {
  if (url === '/api/scaffold') return { artifacts: ARTIFACTS };
  if (url === '/api/scaffold/untracked') return { untracked: [] };
  return { content: '', framework_content: '' };
});

vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/api.js', () => ({
  fetchJSON: (/** @type {string} */ url) => fetchJSON(url),
  apiRequest: vi.fn(async () => ({})),
}));

const gate = vi.hoisted(() => ({ proceed: true }));
const openDrawerTab = vi.fn(async (/** @type {string} */ tabId) => {
  if (!gate.proceed) return false;
  document.getElementById(tabId)?.dispatchEvent(new CustomEvent('drawer:tab-activate'));
  return true;
});

vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/settings.js', () => ({
  openDrawerTab: (/** @type {string} */ tabId) => openDrawerTab(tabId),
}));

const { initScaffoldGallery, listGalleryEntries, revealArtifact, scaffoldGalleryMounted } =
  await import('../../../../src/osprey/interfaces/web_terminal/static/js/scaffold-gallery.js');

function mountDrawer() {
  document.body.innerHTML = `
    <div id="settings-drawer">
      <button class="drawer-tab" data-tab="tab-behavior">Behavior</button>
      <button class="drawer-tab" data-tab="tab-safety">Safety</button>
      <div id="tab-behavior"></div>
      <div id="tab-safety"></div>
    </div>`;
  const drawer = /** @type {any} */ (document.getElementById('settings-drawer'));
  drawer.registerUnsavedGuard = () => {};
  initScaffoldGallery();
}

/** @param {string} tabId */
function detailView(tabId) {
  return /** @type {HTMLElement} */ (document.querySelector(`#${tabId} .prompts-detail-view`));
}

describe('scaffold gallery palette seams', () => {
  beforeEach(() => {
    gate.proceed = true;
    openDrawerTab.mockClear();
    mountDrawer();
  });

  it('lists the files a mounted gallery shows, with their tab and card label', async () => {
    expect(scaffoldGalleryMounted()).toBe(true);
    const entries = await listGalleryEntries();
    // mcp-json belongs to the Config gallery, which this drawer does not carry.
    expect(entries.map((e) => e.name)).toEqual([
      'rules/safety',
      'code-review/rules',
      'hooks/approval',
    ]);
    expect(entries[0]).toMatchObject({
      label: 'safety',
      category: 'instructions',
      tabId: 'tab-behavior',
      tabLabel: 'Behavior',
    });
    expect(entries[1]).toEqual({
      name: 'code-review/rules',
      label: 'rules',
      category: 'code-review',
      tabId: 'tab-safety',
      tabLabel: 'Safety',
      description: "The code reviewer's instructions",
    });
  });

  it('reveal opens the tab through the gate and the file in Preview once loaded', async () => {
    expect(detailView('tab-safety').style.display).toBe('none');
    await revealArtifact('code-review/rules', 'tab-safety');
    expect(openDrawerTab).toHaveBeenCalledWith('tab-safety');
    expect(detailView('tab-safety').style.display).toBe('');
    expect(document.querySelector('#tab-safety .prompts-detail-name')?.textContent).toContain(
      'rules'
    );
    expect(document.querySelector('#tab-safety .prompts-mode-btn.active')?.textContent).toMatch(
      /Preview/
    );
  });

  it('a declined gate opens nothing', async () => {
    gate.proceed = false;
    await revealArtifact('code-review/rules', 'tab-safety');
    expect(detailView('tab-safety').style.display).toBe('none');
  });

  it('an unknown file leaves the tab on its card grid', async () => {
    await revealArtifact('code-review/missing', 'tab-safety');
    expect(openDrawerTab).toHaveBeenCalledWith('tab-safety');
    expect(detailView('tab-safety').style.display).toBe('none');
  });
});
