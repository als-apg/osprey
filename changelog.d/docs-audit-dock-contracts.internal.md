The audit-trail and dock-engine contract pages are now held to the code by
`tests/docs/test_audit_trail_page.py` and `tests/docs/test_dock_engine_page.py`:
the record and tool-call field tables, the surface list in both directions
(including the hook surfaces the hooks compose from their own names), the
posture, decision and login-refusal words, and the vendored engine files,
manifest entries, layout rules and case copies the dock-engine page names. The
dock-engine page's holdings are now one table (what / pinned by / checked by)
and it says where `vendor_manifest.json` lives.
