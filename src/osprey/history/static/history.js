// The history page's one behaviour: the session list's source filter.
// Without this script the list shows every source and the filter stays hidden.
(function () {
  'use strict';

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

  document.addEventListener('DOMContentLoaded', function () {
    const label = document.querySelector('label.source-filter');
    if (!label) return;
    const select = label.querySelector('select');
    label.hidden = false;
    select.addEventListener('change', function () {
      applyFilter(select);
    });
    applyFilter(select);
  });
})();
