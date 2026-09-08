Generated `.claude/settings.json` files no longer make the OSPREY agent print
two "Permission allow rule ... NotebookEdit(...) is not matched by file
permission checks" warnings at every start. The agent-data notebook and
artifact trees are now allowed with `Edit(<path>/**)` rules, the form Claude
Code consults for every file-editing tool. Rebuild a deployment to pick up the
change.
