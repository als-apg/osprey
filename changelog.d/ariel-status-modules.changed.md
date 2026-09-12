`osprey ariel status` reports one enhancement-module table: every registered
module, with whether it is enabled and how many entries it has completed,
failed and left pending. Store keys that no registered module claims are
reported separately, under `orphaned_enhancement_modules` in the `--json`
payload and as one line in the human output, so leftover rows are visible
rather than counted into a module that no longer exists.
