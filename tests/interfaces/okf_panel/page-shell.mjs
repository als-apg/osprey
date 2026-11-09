// @ts-check
/**
 * The KNOWLEDGE panel's real page shell, for the tests that boot app.js or
 * its modules: the `#app` element of static/index.html, so a test runs on the
 * markup the panel ships (the toolbar, the contents list, the reader) and not
 * on a copy that can drift from it.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const INDEX_HTML = join(
  import.meta.dirname,
  '../../../src/osprey/interfaces/okf_panel/static/index.html'
);

/**
 * Replace the document's body with the panel's `#app`. The page is parsed in
 * a <template>, which loads none of its stylesheets or scripts.
 */
export function mountShell() {
  const html = readFileSync(INDEX_HTML, 'utf8');
  const template = document.createElement('template');
  template.innerHTML = html.slice(
    html.indexOf('<body>') + '<body>'.length,
    html.indexOf('</body>')
  );
  const app = template.content.querySelector('#app');
  if (!app) throw new Error('okf_panel index.html has no #app');
  document.body.replaceChildren(app);
}
