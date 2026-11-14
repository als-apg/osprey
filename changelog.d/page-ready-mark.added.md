Every page OSPREY serves sets `data-page-ready="true"` on its `<html>` once it
can be driven, written by one module (`design-system/js/page-ready.js`); a
browser test, a docs recording or a facility script waits for that attribute.
It replaces the hub's `data-panels-ready` and a popped-out window's
`data-window-ready`.
