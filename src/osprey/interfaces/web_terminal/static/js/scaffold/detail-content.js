// @ts-check
/**
 * OSPREY Web Terminal — Scaffold Gallery: detail-view content renderers
 *
 * The two read-only detail-view content renderers: Preview (rendered
 * markdown / highlighted code, plus the structured settings.json and
 * mcp.json views) and Diff (unified diff between the active and
 * framework-default layers, grouped into hunks and word-diffed per pair
 * via diff-utils). Kept separate from scaffold/detail.js to hold both
 * modules comfortably under the 450-line cap -- Preview/Diff is the
 * natural "content rendering" seam, distinct from detail.js's "shell"
 * concern (mode switching, header, dispatch).
 *
 * Mirrors the same factory/injection pattern as the rest of the scaffold
 * module split: {@link createScaffoldGalleryDetailContent} is bound to a
 * fixed gallery host.
 *
 * @module scaffold/detail-content
 */

import { fetchJSON } from '../api.js';
import { tokenize, computeWordDiff, groupChangeBlocks, renderWordsIntoLine } from '../diff-utils.js';
import { renderSettingsJsonEditor, renderMcpJson } from '../config-renderers.js';
import {
  parseFrontMatter,
  extractPythonDocstringFrontMatter,
  renderHighlightedCode,
  renderFlowDiagram,
  renderSourceToggle,
  renderFrontMatterTable,
} from './utils.js';

/**
 * `detailContentEl` grows a `_settingsEditor` property when the settings.json
 * structured editor is mounted into it (see renderSettingsJsonEditor in
 * config-renderers.js) -- read back by ArtifactGallery.saveOverride() (a
 * thin delegator on the gallery instance into scaffold/edit.js).
 * @typedef {HTMLElement & {
 *   _settingsEditor?: { getData(): string, isDirty(): boolean } | null,
 * }} DetailContentElement
 */

/**
 * The subset of an ArtifactGallery instance these content renderers read,
 * write, or call into.
 * @typedef {object} ScaffoldGalleryDetailContentHost
 * @property {any} selectedArtifact
 * @property {boolean} editDirty
 * @property {DetailContentElement|null} detailContentEl
 * @property {() => void} renderDetailModes
 */

/**
 * Create the scaffold gallery's detail-content renderers, bound to a fixed
 * gallery host.
 *
 * @param {ScaffoldGalleryDetailContentHost} gallery
 */
export function createScaffoldGalleryDetailContent(gallery) {
  /** @returns {Promise<void>} */
  async function renderPreview() {
    const data = await fetchJSON(`/api/scaffold/${encodeURIComponent(gallery.selectedArtifact.name)}`); // fetchJSON prefixes internally
    const content = data.content || '';
    const language = data.language || gallery.selectedArtifact.language || 'text';
    const artifactName = gallery.selectedArtifact.name || '';

    if (!gallery.detailContentEl) return;
    gallery.detailContentEl.innerHTML = '';

    const wrapper = document.createElement('div');
    wrapper.className = 'prompts-preview-content';

    // settings-json: use interactive editor directly in preview mode
    if (artifactName === 'settings-json' && language === 'json') {
      const structured = renderSettingsJsonEditor(content, (isDirty) => {
        gallery.editDirty = isDirty;
        gallery.renderDetailModes();
      });
      if (structured) {
        // Attach _settingsEditor API on detailContentEl for saveOverride()
        gallery.detailContentEl._settingsEditor = structured._settingsEditor;
        wrapper.appendChild(structured);
        wrapper.appendChild(renderSourceToggle(content, 'json'));
        gallery.detailContentEl.appendChild(wrapper);
        return;
      }
    }

    if (artifactName === 'mcp-json' && language === 'json') {
      const structured = renderMcpJson(content);
      if (structured) {
        wrapper.appendChild(structured);
        wrapper.appendChild(renderSourceToggle(content, 'json'));
        gallery.detailContentEl.appendChild(wrapper);
        return;
      }
    }

    if (language === 'markdown') {
      const { frontMatter, body } = parseFrontMatter(content);

      if (frontMatter) {
        wrapper.appendChild(renderFrontMatterTable(frontMatter));
      }

      const mdDiv = document.createElement('div');
      mdDiv.className = 'osprey-md-rendered';
      if (typeof marked !== 'undefined') {
        try {
          mdDiv.innerHTML = marked.parse(body);
        } catch {
          mdDiv.textContent = body;
        }
      } else {
        mdDiv.textContent = body;
      }
      wrapper.appendChild(mdDiv);
    } else if (language === 'python') {
      const parsed = extractPythonDocstringFrontMatter(content);

      if (parsed.frontMatter) {
        wrapper.appendChild(renderFrontMatterTable(parsed.frontMatter));

        if (parsed.flowDiagram) {
          wrapper.appendChild(renderFlowDiagram(parsed.flowDiagram));
        }

        if (parsed.body) {
          const mdDiv = document.createElement('div');
          mdDiv.className = 'osprey-md-rendered';
          if (typeof marked !== 'undefined') {
            try {
              mdDiv.innerHTML = marked.parse(parsed.body);
            } catch {
              mdDiv.textContent = parsed.body;
            }
          } else {
            mdDiv.textContent = parsed.body;
          }
          wrapper.appendChild(mdDiv);
        }

        wrapper.appendChild(renderSourceToggle(parsed.sourceCode, 'python'));
      } else {
        wrapper.appendChild(renderHighlightedCode(content, language));
      }
    } else {
      wrapper.appendChild(renderHighlightedCode(content, language));
    }

    gallery.detailContentEl.appendChild(wrapper);
  }

  /** @returns {Promise<void>} */
  async function renderDiff() {
    const data = await fetchJSON(
      `/api/scaffold/${encodeURIComponent(gallery.selectedArtifact.name)}/diff` // fetchJSON prefixes internally
    );

    if (!gallery.detailContentEl) return;
    gallery.detailContentEl.innerHTML = '';

    if (!data.has_diff) {
      gallery.detailContentEl.innerHTML = '<div class="prompts-no-diff">No differences from framework default.</div>';
      return;
    }

    const stats = document.createElement('div');
    stats.className = 'prompts-diff-stats';
    stats.innerHTML =
      `<span class="prompts-diff-add">+${data.additions}</span> ` +
      `<span class="prompts-diff-del">−${data.deletions}</span>`;
    gallery.detailContentEl.appendChild(stats);

    const diffBlock = document.createElement('div');
    diffBlock.className = 'prompts-diff-block';

    const rawLines = (data.unified_diff || '').split('\n');
    const blocks = groupChangeBlocks(rawLines);

    for (const block of blocks) {
      if (block.type === 'change') {
        // groupChangeBlocks only sets delLines/addLines on 'change' blocks --
        // narrowed by block.type above, but its return type doesn't encode
        // that relationship, hence the assertions.
        const delLines = /** @type {string[]} */ (block.delLines);
        const addLines = /** @type {string[]} */ (block.addLines);
        const paired = Math.min(delLines.length, addLines.length);

        // Paired lines: compute word diff per pair
        for (let k = 0; k < paired; k++) {
          const oldTokens = tokenize(delLines[k]);
          const newTokens = tokenize(addLines[k]);
          const ops = computeWordDiff(oldTokens, newTokens);

          const delEl = document.createElement('div');
          delEl.className = 'prompts-diff-line del';
          renderWordsIntoLine(delEl, delLines[k], ops, 'del');
          diffBlock.appendChild(delEl);

          const addEl = document.createElement('div');
          addEl.className = 'prompts-diff-line add';
          renderWordsIntoLine(addEl, addLines[k], ops, 'add');
          diffBlock.appendChild(addEl);
        }

        // Surplus unpaired lines: plain textContent
        for (let k = paired; k < delLines.length; k++) {
          const el = document.createElement('div');
          el.className = 'prompts-diff-line del';
          el.textContent = delLines[k];
          diffBlock.appendChild(el);
        }
        for (let k = paired; k < addLines.length; k++) {
          const el = document.createElement('div');
          el.className = 'prompts-diff-line add';
          el.textContent = addLines[k];
          diffBlock.appendChild(el);
        }
      } else {
        // context, hunk, unpaired del, unpaired add -- groupChangeBlocks
        // always sets `lines` on non-'change' blocks.
        const lines = /** @type {string[]} */ (block.lines);
        for (const line of lines) {
          const lineEl = document.createElement('div');
          lineEl.className = 'prompts-diff-line';

          if (block.type === 'hunk') lineEl.classList.add('hunk');
          else if (block.type === 'del') lineEl.classList.add('del');
          else if (block.type === 'add') lineEl.classList.add('add');
          else lineEl.classList.add('context');

          lineEl.textContent = line;
          diffBlock.appendChild(lineEl);
        }
      }
    }

    gallery.detailContentEl.appendChild(diffBlock);
  }

  return { renderPreview, renderDiff };
}
