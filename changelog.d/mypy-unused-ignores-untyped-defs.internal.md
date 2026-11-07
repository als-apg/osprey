The type check refuses a `# type: ignore` that suppresses nothing, and it checks
the bodies of unannotated functions too. The 27 stale suppressions are gone, and
the 44 errors those bodies hid are fixed in the code. One fix changes
behaviour: an `additional_dirs` entry that names no `src` is skipped, where a dict
with neither `src` nor `dst` used to crash the build with a `TypeError`, and a
null `dst` now falls back to `src`.
