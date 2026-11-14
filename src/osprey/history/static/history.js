// The history page's behaviour: the session list's source filter, then the
// page-ready mark.
// Without this script the list shows every source and the filter stays hidden.
(function () {
  'use strict';

  // The page-ready writer is a design-system module, served beside this
  // script's own root; a classic script reaches it through import().
  const readyModule = new URL('../design-system/js/page-ready.js', document.currentScript.src).href;

  function applyFilter(select) {
    const wanted = select.value;
    const list = document.querySelector('ol.sessions');
    if (!list) return;
    const visibleDays = {};
    list.querySelectorAll('li.session').forEach(function (row) {
      const shown = !wanted || row.dataset.source === wanted;
      row.hidden = !shown;
      if (shown) visibleDays[row.dataset.day] = true;
    });
    list.querySelectorAll('li.day').forEach(function (day) {
      day.hidden = !visibleDays[day.dataset.day];
    });
  }

  function wireFilter() {
    const label = document.querySelector('label.source-filter');
    if (!label) return;
    const select = label.querySelector('select');
    label.hidden = false;
    select.addEventListener('change', function () {
      applyFilter(select);
    });
    applyFilter(select);
  }

  document.addEventListener('DOMContentLoaded', function () {
    wireFilter();
    // The page is ready once the filter is wired (base.html declares the mark
    // pending until then).
    import(readyModule).then(function (ready) {
      ready.markPageReady();
    });
  });
})();
