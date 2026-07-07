// @ts-check
/**
 * OSPREY Web Terminal — Scaffold Gallery: card templates
 *
 * The per-artifact card template (renderArtifactCard) and its skill-group
 * variant (renderSkillGroup — a multi-artifact card with a picker, for
 * skills that bundle more than one file). Kept separate from
 * scaffold/view.js to hold every module under the 450-line cap; the two
 * card templates are the natural "one concern" seam since they're the only
 * pieces of the view layer that don't touch the gallery's
 * search/filter/category-grid state.
 *
 * Mirrors the same factory/injection pattern as the rest of the scaffold
 * module split: {@link createScaffoldGalleryCards} is bound to the gallery
 * host's `openDetail` callback — the only gallery state these templates
 * need, since everything else (name, description, status, etc.) comes
 * from the `artifact` argument each call already carries.
 *
 * @module scaffold/cards
 */

import { iconForCategory } from './utils.js';

/**
 * The subset of an ArtifactGallery instance these card templates call into.
 * @typedef {object} ScaffoldGalleryCardHost
 * @property {(artifact: any) => void} openDetail
 */

/**
 * Create the scaffold gallery's card-template renderers, bound to a fixed
 * gallery host's openDetail callback.
 *
 * @param {ScaffoldGalleryCardHost} gallery
 */
export function createScaffoldGalleryCards(gallery) {
  /**
   * @param {HTMLElement} section
   * @param {any} artifact
   * @param {string} cat
   * @returns {void}
   */
  function renderArtifactCard(section, artifact, cat) {
    const card = document.createElement('div');
    card.className = 'prompts-card';
    card.dataset.name = artifact.name;

    const icon = document.createElement('div');
    icon.className = 'prompts-card-icon';
    icon.textContent = iconForCategory(cat);

    const body = document.createElement('div');
    body.className = 'prompts-card-body';

    const nameEl = document.createElement('div');
    nameEl.className = 'prompts-card-name';
    const displayName = artifact.name.includes('/')
      ? artifact.name.split('/').slice(1).join('/')
      : artifact.name;
    nameEl.textContent = displayName;

    const descEl = document.createElement('div');
    descEl.className = 'prompts-card-desc';
    descEl.textContent = artifact.summary || artifact.description || '';

    body.appendChild(nameEl);
    body.appendChild(descEl);

    const badge = document.createElement('span');
    const owned = artifact.status === 'user-owned';
    badge.className = `prompts-badge ${owned ? 'user-owned' : 'framework'}`;
    badge.textContent = owned ? 'PROJECT-OWNED' : 'FRAMEWORK';

    card.appendChild(icon);
    card.appendChild(body);
    card.appendChild(badge);

    card.addEventListener('click', () => gallery.openDetail(artifact));
    section.appendChild(card);
  }

  /**
   * @param {HTMLElement} section
   * @param {string} skillName
   * @param {any[]} groupArtifacts
   * @returns {void}
   */
  function renderSkillGroup(section, skillName, groupArtifacts) {
    if (groupArtifacts.length === 1) {
      renderArtifactCard(section, groupArtifacts[0], 'skills');
      return;
    }

    const sorted = [...groupArtifacts].sort((a, b) => {
      const aDepth = a.name.split('/').length;
      const bDepth = b.name.split('/').length;
      return aDepth - bDepth || a.name.localeCompare(b.name);
    });

    const card = document.createElement('div');
    card.className = 'prompts-card prompts-skill-group';
    card.dataset.name = sorted[0].name;

    let selectedArt = sorted[0];

    const icon = document.createElement('div');
    icon.className = 'prompts-card-icon';
    icon.textContent = iconForCategory('skills');

    const body = document.createElement('div');
    body.className = 'prompts-card-body';

    const nameEl = document.createElement('div');
    nameEl.className = 'prompts-card-name';
    nameEl.textContent = skillName;

    const descEl = document.createElement('div');
    descEl.className = 'prompts-card-desc';
    descEl.textContent = selectedArt.summary || selectedArt.description || '';

    body.appendChild(nameEl);
    body.appendChild(descEl);

    const select = document.createElement('select');
    select.className = 'prompts-skill-select';
    for (const art of sorted) {
      const opt = document.createElement('option');
      opt.value = art.name;
      opt.textContent = (art.output_path || art.name).split('/').pop();
      select.appendChild(opt);
    }
    select.addEventListener('click', (e) => e.stopPropagation());
    select.addEventListener('change', (e) => {
      const value = /** @type {HTMLSelectElement} */ (e.target).value;
      selectedArt = sorted.find((a) => a.name === value) || sorted[0];
      descEl.textContent = selectedArt.summary || selectedArt.description || '';
    });
    body.appendChild(select);

    const badge = document.createElement('span');
    const ownedSkill = selectedArt.status === 'user-owned';
    badge.className = `prompts-badge ${ownedSkill ? 'user-owned' : 'framework'}`;
    badge.textContent = ownedSkill ? 'PROJECT-OWNED' : 'FRAMEWORK';

    card.appendChild(icon);
    card.appendChild(body);
    card.appendChild(badge);

    card.addEventListener('click', () => gallery.openDetail(selectedArt));
    section.appendChild(card);
  }

  return { renderArtifactCard, renderSkillGroup };
}
