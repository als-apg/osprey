The profile parser's positive-integer checks and the model sidecar's
absolute-path check read the shared config guards instead of spelling their own
copy, so every surface refuses the same values for the same reason.
