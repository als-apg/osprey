A dispatched run no longer loses a subagent's report: the dispatch tool policy lets a declared subagent call `SubagentHandback`, which the permission hook used to refuse as outside its `tools:` list.
