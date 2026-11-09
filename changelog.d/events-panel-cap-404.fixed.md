Web terminal: opening the events panel, or any panel declared under
`web.panels`, no longer logs a 404 for `/api/panels/<id>/cap` in the browser
console. The hub now answers that a panel is first-party instead of returning
404, and treats a 404 as an opaque panel with no cap.
